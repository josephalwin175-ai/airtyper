"""AirTyper Link: pair AirTypers on the same network and throw items between them.

  Discovery   UDP beacons every 1.5 s (broadcast, plus unicast to manual addresses and to anyone who
              beacons us, so one working direction is enough).
  Session     one TCP connection per paired device. Both sides send a hello with a long-term X25519
              key and a fresh one; the session keys come from both Diffie-Hellman results, so only the
              real owner of a pinned key can talk to us. Frames are ChaCha20-Poly1305 encrypted.
  Pairing     the first connection is unpinned: both screens show a 6-digit code derived from all four
              public keys. A man in the middle would make the codes differ. When both people confirm,
              each side pins the other's key (stored in %APPDATA%\\AirTyper\\link.json).
  Items       text, images (PNG) and files/folders, streamed in chunks on a low-priority queue so
              ghost hands and control messages never wait behind a big file.

Runs on background threads. The app calls the public methods from its main thread and drains
events with poll(). No Windows-specific calls, so it can be tested on its own: python link.py --selftest
"""
import base64
import hashlib
import json
import os
import queue
import socket
import struct
import threading
import time
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

UDP_PORT = 47820
TCP_PORT = 47821
BEACON_EVERY = 1.5          # s
PEER_TIMEOUT = 6.0          # s without a beacon before a device counts as gone
DIAL_EVERY = 3.0            # s between reconnect attempts to a paired device
PAIR_TIMEOUT = 90.0         # s a pairing may stay unfinished
PING_EVERY = 4.0            # s of silence before a keep-alive
READ_TIMEOUT = 15.0         # s of silence before a session counts as dead
CHUNK = 256 * 1024
MAX_FRAME = 2 * 1024 * 1024
MAX_ITEM = 512 * 1024 * 1024
PROTO = "airtyper-link"
SIDES = ("right", "left", "top")


def b64(b):
    return base64.b64encode(b).decode()


def unb64(s):
    return base64.b64decode(s)


def pub_bytes(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def local_ipv4s():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # the address used for the default route (works even when the hostname lookup is thin)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def broadcast_addrs():
    """Directed broadcast address of every real IPv4 network we are on (255.255.255.255 only leaves
    through one adapter on Windows, often a virtual one)."""
    out = set()
    try:
        import ipaddress
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and a.netmask and not a.address.startswith(("127.", "169.254.")):
                    out.add(str(ipaddress.IPv4Network(f"{a.address}/{a.netmask}", strict=False).broadcast_address))
    except Exception:
        out = {ip.rsplit(".", 1)[0] + ".255" for ip in local_ipv4s()}  # assume a /24 home network
    return out


def safe_rel(rel):
    """A received relative path, or None if it tries to escape the download folder."""
    parts = rel.replace("\\", "/").split("/")
    if not parts or any(p in ("", ".", "..") or ":" in p for p in parts):
        return None
    return os.path.join(*parts)


class Identity:
    """Long-term key + the devices we trust. Lives outside the repo, in the user's profile."""

    def __init__(self, path):
        self.path = path
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        self.id = data.get("id") or uuid.uuid4().hex[:16]
        raw = data.get("key")
        self.key = X25519PrivateKey.from_private_bytes(unb64(raw)) if raw else X25519PrivateKey.generate()
        self.pinned = data.get("pinned", {})  # peer id -> {"name", "static", "side"}
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        raw = self.key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                     serialization.NoEncryption())
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"id": self.id, "key": b64(raw), "pinned": self.pinned}, f, indent=2)
        os.replace(tmp, self.path)


class LinkError(Exception):
    pass


def _send_raw(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(min(n - len(buf), 1 << 20))
        if not part:
            raise ConnectionError("closed")
        buf += part
    return bytes(buf)


def _recv_raw(sock):
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    if n > MAX_FRAME + 4096:
        raise LinkError("frame too large")
    return _recv_exact(sock, n)


class OutJob:
    """An item being sent: yields its header, then chunks, one per call, so other messages interleave."""

    def __init__(self, tid, meta, blobs):
        self.tid, self.meta, self.blobs = tid, meta, blobs  # blobs: list of (rel name, bytes | path, size)
        self.total = sum(b[2] for b in blobs)
        self.sent = 0
        self.started = False
        self.index = self.off = 0
        self.fh = None
        self.cancelled = False

    def next(self):
        if self.cancelled:
            return None
        if not self.started:
            self.started = True
            hdr = dict(self.meta, t="item", id=self.tid, total=self.total,
                       files=[[b[0], b[2]] for b in self.blobs])
            return hdr, b""
        while self.index < len(self.blobs):
            fi = self.index
            _, src, size = self.blobs[fi]
            want = min(CHUNK, size - self.off)
            if want <= 0:
                data = b""
            elif isinstance(src, (bytes, bytearray)):
                data = bytes(src[self.off:self.off + want])
            else:
                if self.fh is None:
                    self.fh = open(src, "rb")
                data = self.fh.read(want)
            self.off += len(data)
            if not data or self.off >= size:  # blob finished (zero-byte files are created by the receiver)
                if self.fh:
                    self.fh.close()
                    self.fh = None
                self.index, self.off = self.index + 1, 0
            if data:
                self.sent += len(data)
                return {"t": "chunk", "id": self.tid, "f": fi}, data
        return None

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


class Session:
    def __init__(self, link, sock, addr, initiator):
        self.link, self.sock, self.addr, self.initiator = link, sock, addr, initiator
        self.peer_id = self.peer_name = self.peer_static = None
        self.trusted = False
        self.alive = True
        self.sas = None
        self.out = queue.PriorityQueue()
        self.seq = 0
        self.tx_n = self.rx_n = 0
        self.last_rx = time.time()
        self.wlock = threading.Lock()

    # ---------- handshake ----------
    def handshake(self):
        ident = self.link.ident
        eph = X25519PrivateKey.generate()
        mine = {"t": "hello", "proto": PROTO, "v": 1, "id": ident.id, "name": self.link.name,
                "static": b64(pub_bytes(ident.key)), "eph": b64(pub_bytes(eph))}
        self.sock.settimeout(10)
        _send_raw(self.sock, json.dumps(mine).encode())
        peer = json.loads(_recv_raw(self.sock))
        if peer.get("proto") != PROTO or not isinstance(peer.get("id"), str) or peer["id"] == ident.id:
            raise LinkError("not an AirTyper peer")
        ps, pe = unb64(peer["static"]), unb64(peer["eph"])
        if len(ps) != 32 or len(pe) != 32:
            raise LinkError("bad keys")
        dh = eph.exchange(X25519PublicKey.from_public_bytes(pe)) + \
            ident.key.exchange(X25519PublicKey.from_public_bytes(ps))
        a = pub_bytes(ident.key) + pub_bytes(eph)
        transcript = hashlib.sha256(b"airtyper-link-v1" + b"".join(sorted([a, ps + pe]))).digest()
        okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=transcript, info=b"session keys").derive(dh)
        lo = ident.id < peer["id"]
        self.tx = ChaCha20Poly1305(okm[:32] if lo else okm[32:])
        self.rx = ChaCha20Poly1305(okm[32:] if lo else okm[:32])
        self.sas = f"{int.from_bytes(hashlib.sha256(b'sas' + transcript).digest()[:4], 'big') % 1_000_000:06d}"
        self.peer_id, self.peer_name, self.peer_static = peer["id"], str(peer.get("name", "?"))[:40], peer["static"]
        pin = ident.pinned.get(self.peer_id)
        if pin:
            if pin["static"] != self.peer_static:
                raise LinkError(f"{self.peer_name} has a different identity than when you paired")
            self.trusted = True
        # key confirmation: a fake peer without the pinned private key fails here
        self._write({"t": "ready"})
        hdr, _ = self._read()
        if hdr.get("t") != "ready":
            raise LinkError("handshake failed")
        self.sock.settimeout(READ_TIMEOUT)

    # ---------- framing ----------
    def _write(self, hdr, payload=b""):
        h = json.dumps(hdr, separators=(",", ":")).encode()
        plain = struct.pack(">I", len(h)) + h + payload
        with self.wlock:
            nonce = struct.pack(">4xQ", self.tx_n)
            self.tx_n += 1
            _send_raw(self.sock, self.tx.encrypt(nonce, plain, None))

    def _read(self):
        data = _recv_raw(self.sock)
        nonce = struct.pack(">4xQ", self.rx_n)
        self.rx_n += 1
        plain = self.rx.decrypt(nonce, data, None)  # raises if tampered or wrong key
        (n,) = struct.unpack(">I", plain[:4])
        self.last_rx = time.time()
        return json.loads(plain[4:4 + n]), plain[4 + n:]

    def send(self, hdr, payload=b"", prio=1):
        """prio 0 = control / ghost hands, 2 = bulk (item jobs)."""
        self.seq += 1
        self.out.put((prio, self.seq, (hdr, payload)))

    def send_job(self, job):
        self.seq += 1
        self.out.put((2, self.seq, job))

    def start(self):
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._writer, daemon=True).start()

    def _writer(self):
        try:
            while self.alive:
                try:
                    prio, seq, item = self.out.get(timeout=PING_EVERY)
                except queue.Empty:
                    self._write({"t": "ping"})
                    continue
                if isinstance(item, OutJob):
                    nxt = item.next()
                    if nxt is None:
                        item.close()
                        if not item.cancelled:
                            self._write({"t": "end", "id": item.tid})
                            self.link._progress(self, item.tid, "out", 1.0, done=True)
                        continue
                    self._write(*nxt)
                    if item.total:
                        self.link._progress(self, item.tid, "out", item.sent / item.total)
                    self.seq += 1
                    self.out.put((2, self.seq, item))  # requeue: higher-priority messages go first
                else:
                    self._write(*item)
        except (OSError, LinkError, ValueError):
            pass
        finally:
            self.close()

    def _reader(self):
        try:
            while self.alive:
                hdr, payload = self._read()
                if hdr.get("t") != "ping":
                    self.link._on_msg(self, hdr, payload)
        except Exception:
            pass
        finally:
            self.close()

    def close(self):
        if not self.alive:
            return
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass
        while True:
            try:
                _, _, item = self.out.get_nowait()
                if isinstance(item, OutJob):
                    item.close()
            except queue.Empty:
                break
        self.link._session_closed(self)


class Link:
    def __init__(self, name, ident_path, recv_dir, udp_port=UDP_PORT, tcp_port=TCP_PORT, manual=()):
        self.name = name[:40]
        self.ident = Identity(ident_path)
        self.recv_dir = recv_dir
        self.udp_port, self.tcp_pref = udp_port, tcp_port
        self.manual = [self._parse_addr(m) for m in manual]
        self.events = queue.Queue()
        self.lock = threading.RLock()
        self.peers = {}       # discovered: id -> {"name", "addr", "port", "seen"}
        self.sessions = {}    # trusted sessions: peer id -> Session
        self.pairing = {}     # peer id -> {"state": waiting|incoming|code, "session", "sas", "me", "them", "t"}
        self.dialing = set()
        self.last_dial = {}
        self.incoming = {}    # transfer id -> receive state
        self.jobs = {}        # transfer id -> OutJob
        self.replied = {}
        self.waiting_since = {}  # paired + online but not connected, since when
        self.running = False
        self.tcp_port = None

    def _parse_addr(self, m):
        host, _, port = str(m).partition(":")
        return host, int(port) if port else self.udp_port

    # ---------- lifecycle ----------
    def start(self):
        self.running = True
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.tcp.bind(("0.0.0.0", self.tcp_pref))
        except OSError:
            self.tcp.bind(("0.0.0.0", 0))
        self.tcp.listen(8)
        self.tcp_port = self.tcp.getsockname()[1]
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.udp.bind(("0.0.0.0", self.udp_port))
        self.udp.settimeout(1.0)
        for fn in (self._beacon_loop, self._udp_loop, self._accept_loop, self._maintain_loop):
            threading.Thread(target=fn, daemon=True).start()

    def stop(self):
        self.running = False
        for s in (getattr(self, "tcp", None), getattr(self, "udp", None)):
            try:
                s and s.close()
            except OSError:
                pass
        with self.lock:
            sessions = list(self.sessions.values()) + [p["session"] for p in self.pairing.values() if p.get("session")]
        for s in sessions:
            s.close()

    def _emit(self, **e):
        self.events.put(e)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    # ---------- discovery ----------
    def _beacon(self):
        return json.dumps({"proto": PROTO, "v": 1, "id": self.ident.id, "name": self.name,
                           "port": self.tcp_port}).encode()

    def _beacon_loop(self):
        bcast, refreshed = set(), 0.0
        while self.running:
            msg = self._beacon()
            if time.time() - refreshed > 15:  # networks change (Wi-Fi switch, hotspot)
                bcast, refreshed = broadcast_addrs(), time.time()
            targets = {("255.255.255.255", self.udp_port)} | {(b, self.udp_port) for b in bcast}
            targets.update(self.manual)
            for t in targets:
                try:
                    self.udp.sendto(msg, t)
                except OSError:
                    pass
            time.sleep(BEACON_EVERY)

    def _udp_loop(self):
        while self.running:
            try:
                data, (ip, port) = self.udp.recvfrom(2048)
                b = json.loads(data)
            except socket.timeout:
                continue
            except (OSError, ValueError):
                if not self.running:
                    return
                continue
            if b.get("proto") != PROTO or b.get("id") == self.ident.id or not isinstance(b.get("port"), int):
                continue
            pid = b["id"]
            with self.lock:
                new = pid not in self.peers
                self.peers[pid] = {"name": str(b.get("name", "?"))[:40], "addr": ip, "port": b["port"],
                                   "seen": time.time()}
                pin = self.ident.pinned.get(pid)
                if pin and pin.get("name") != self.peers[pid]["name"]:
                    pin["name"] = self.peers[pid]["name"]
            if new:
                self._emit(type="peers")
            # answer directly, so discovery works even if only one direction of broadcast gets through
            if time.time() - self.replied.get(pid, 0) > BEACON_EVERY:
                self.replied[pid] = time.time()
                try:
                    self.udp.sendto(self._beacon(), (ip, port))
                except OSError:
                    pass

    # ---------- connections ----------
    def _accept_loop(self):
        while self.running:
            try:
                sock, addr = self.tcp.accept()
            except OSError:
                return
            threading.Thread(target=self._setup, args=(sock, addr, False), daemon=True).start()

    def _dial(self, pid, then_pair=False):
        with self.lock:
            p = self.peers.get(pid)
            if not p or pid in self.dialing:
                return
            self.dialing.add(pid)
            self.last_dial[pid] = time.time()
        try:
            sock = socket.create_connection((p["addr"], p["port"]), timeout=4)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._setup(sock, (p["addr"], p["port"]), True, then_pair)
        except OSError as e:
            if then_pair:
                with self.lock:
                    self.pairing.pop(pid, None)
                self._emit(type="pair_failed", peer=pid, reason=f"Could not reach {p['name']} ({e.__class__.__name__})")
        finally:
            with self.lock:
                self.dialing.discard(pid)

    def _setup(self, sock, addr, initiator, then_pair=False):
        s = Session(self, sock, addr, initiator)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.handshake()
        except Exception as e:
            try:
                sock.close()
            except OSError:
                pass
            if isinstance(e, LinkError) and "identity" in str(e):
                self._emit(type="error", message=str(e))
            if then_pair:
                self._emit(type="pair_failed", peer=s.peer_id, reason="Handshake failed")
            return
        if s.trusted:
            self._register(s)
            return
        with self.lock:  # unpinned: only good for pairing
            st = self.pairing.get(s.peer_id)
            if initiator and then_pair:
                self.pairing[s.peer_id] = {"state": "waiting", "session": s, "sas": s.sas,
                                           "me": False, "them": False, "t": time.time(), "name": s.peer_name}
        s.start()
        if initiator and then_pair:
            s.send({"t": "pair_req"}, prio=0)
            self._emit(type="pairing", peer=s.peer_id)
        elif st is None:
            threading.Timer(30, lambda: (not self._pairing_session(s)) and s.close()).start()

    def _pairing_session(self, s):
        with self.lock:
            p = self.pairing.get(s.peer_id)
            return bool(p and p.get("session") is s)

    def _register(self, s):
        with self.lock:
            old = self.sessions.get(s.peer_id)
            if old and old.alive:
                # both sides dialled at once: keep the connection started by the lower id, on both ends
                low = min(self.ident.id, s.peer_id)
                keep_new = (self.ident.id if s.initiator else s.peer_id) == low
                if not keep_new:
                    s.close()
                    return
                self.sessions[s.peer_id] = s
                old_to_close = old
            else:
                old_to_close = None
                self.sessions[s.peer_id] = s
        if old_to_close:
            old_to_close.alive = False
            try:
                old_to_close.sock.close()
            except OSError:
                pass
        s.start()
        self._emit(type="connected", peer=s.peer_id, name=s.peer_name)

    def _session_closed(self, s):
        with self.lock:
            was = self.sessions.get(s.peer_id) is s
            if was:
                del self.sessions[s.peer_id]
            p = self.pairing.get(s.peer_id)
            pair_lost = bool(p and p.get("session") is s and p["state"] != "done")
            if pair_lost:
                del self.pairing[s.peer_id]
            for tid, st in list(self.incoming.items()):
                if st["session"] is s:
                    self._abort_incoming(tid)
                    self._emit(type="transfer_failed", id=tid, peer=s.peer_id)
        if was:
            self._emit(type="disconnected", peer=s.peer_id, name=s.peer_name)
        if pair_lost:
            self._emit(type="pair_failed", peer=s.peer_id, reason=f"Pairing with {s.peer_name} stopped")

    def _maintain_loop(self):
        while self.running:
            now = time.time()
            dial = []
            with self.lock:
                for pid in [k for k, p in self.peers.items() if now - p["seen"] > PEER_TIMEOUT]:
                    del self.peers[pid]
                    self._emit(type="peers")
                for pid, p in list(self.pairing.items()):
                    if now - p["t"] > PAIR_TIMEOUT:
                        if p.get("session"):
                            p["session"].close()
                        self.pairing.pop(pid, None)
                        self._emit(type="pair_failed", peer=pid, reason="Pairing timed out")
                for pid in self.ident.pinned:
                    if pid not in self.peers or pid in self.sessions:
                        self.waiting_since.pop(pid, None)
                        continue
                    since = self.waiting_since.setdefault(pid, now)
                    if pid not in self.dialing and now - self.last_dial.get(pid, 0) > DIAL_EVERY:
                        # the lower id dials first; the other side joins in if that keeps failing
                        if self.ident.id < pid or now - since > 2 * DIAL_EVERY:
                            dial.append(pid)
            for pid in dial:
                threading.Thread(target=self._dial, args=(pid,), daemon=True).start()
            time.sleep(1.0)

    # ---------- pairing (public) ----------
    def request_pair(self, pid):
        with self.lock:
            if pid in self.ident.pinned or pid in self.pairing or pid not in self.peers:
                return
        threading.Thread(target=self._dial, args=(pid, True), daemon=True).start()

    def accept_pair(self, pid):
        with self.lock:
            p = self.pairing.get(pid)
            if not p or p["state"] != "incoming":
                return
            p["state"], p["t"] = "code", time.time()
            s = p["session"]
        s.send({"t": "pair_ok"}, prio=0)
        self._emit(type="pair_code", peer=pid, sas=p["sas"])

    def confirm_pair(self, pid):
        with self.lock:
            p = self.pairing.get(pid)
            if not p or p["state"] != "code" or p["me"]:
                return
            p["me"] = True
            s = p["session"]
        s.send({"t": "pair_confirm"}, prio=0)
        self._maybe_finish(pid)

    def cancel_pair(self, pid):
        with self.lock:
            p = self.pairing.pop(pid, None)
        if p and p.get("session"):
            p["session"].send({"t": "pair_cancel"}, prio=0)
            threading.Timer(0.3, p["session"].close).start()
        self._emit(type="pair_failed", peer=pid, reason="Cancelled")

    def _maybe_finish(self, pid):
        with self.lock:
            p = self.pairing.get(pid)
            if not p or not (p["me"] and p["them"]):
                return
            s = p["session"]
            used = {v.get("side") for v in self.ident.pinned.values()}
            side = next((x for x in SIDES if x not in used), "right")
            self.ident.pinned[pid] = {"name": s.peer_name, "static": s.peer_static, "side": side}
            self.ident.save()
            p["state"] = "done"
            del self.pairing[pid]
            s.trusted = True
            old = self.sessions.get(pid)
            self.sessions[pid] = s
        if old and old is not s:
            old.close()
        self._emit(type="paired", peer=pid, name=s.peer_name)
        self._emit(type="connected", peer=pid, name=s.peer_name)

    def unpair(self, pid):
        with self.lock:
            self.ident.pinned.pop(pid, None)
            self.ident.save()
            s = self.sessions.pop(pid, None)
        if s:
            s.send({"t": "unpaired"}, prio=0)
            threading.Timer(0.3, s.close).start()
        self._emit(type="peers")

    def set_side(self, pid, side):
        with self.lock:
            if pid in self.ident.pinned and side in SIDES:
                self.ident.pinned[pid]["side"] = side
                self.ident.save()

    # ---------- state for the UI ----------
    def snapshot(self):
        with self.lock:
            out = []
            ids = list(dict.fromkeys(list(self.ident.pinned) + list(self.peers) + list(self.pairing)))
            for pid in ids:
                pin, peer, pr = self.ident.pinned.get(pid), self.peers.get(pid), self.pairing.get(pid)
                name = (pin or peer or pr or {}).get("name", "?")
                out.append({"id": pid, "name": name, "online": peer is not None, "paired": pin is not None,
                            "connected": pid in self.sessions, "side": pin.get("side") if pin else None,
                            "pair": pr["state"] if pr else None, "sas": pr["sas"] if pr else None,
                            "confirmed": bool(pr and pr["me"])})
            out.sort(key=lambda d: (not d["connected"], not d["paired"], d["name"].lower()))
            return out

    def connected(self):
        with self.lock:
            return [(pid, s.peer_name, self.ident.pinned.get(pid, {}).get("side", "right"))
                    for pid, s in self.sessions.items() if s.trusted]

    # ---------- items (public) ----------
    def send_item(self, pid, item):
        """item: {"kind": "text", "text"} | {"kind": "image", "png"} | {"kind": "files", "paths"}"""
        with self.lock:
            s = self.sessions.get(pid)
        if not s or not s.trusted:
            return None
        tid = uuid.uuid4().hex[:12]
        kind = item["kind"]
        if kind == "text":
            data = item["text"].encode("utf-8")
            blobs = [("text", data, len(data))]
        elif kind == "image":
            blobs = [("image.png", item["png"], len(item["png"]))]
        else:
            blobs = []
            for path in item["paths"]:
                base = os.path.basename(os.path.normpath(path))
                if os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for fn in files:
                            full = os.path.join(root, fn)
                            rel = os.path.join(base, os.path.relpath(full, path)).replace("\\", "/")
                            blobs.append((rel, full, os.path.getsize(full)))
                elif os.path.isfile(path):
                    blobs.append((base, path, os.path.getsize(path)))
            if not blobs:
                return None
        if sum(b[2] for b in blobs) > MAX_ITEM:
            self._emit(type="error", message="Too big to throw (limit 512 MB)")
            return None
        meta = {"kind": kind, "label": item.get("label", "")[:200]}
        job = OutJob(tid, meta, blobs)
        with self.lock:
            self.jobs[tid] = job
        s.send_job(job)
        self._emit(type="progress", id=tid, peer=pid, dir="out", frac=0.0, done=False)
        return tid

    def send_ghost(self, hands, holding):
        with self.lock:
            sessions = [s for s in self.sessions.values() if s.trusted]
        for s in sessions:
            if s.out.qsize() < 8:  # never pile up stale hands behind a slow link
                s.send({"t": "ghost", "h": hands, "hold": holding}, prio=0)

    def taken(self, pid, tid):
        with self.lock:
            s = self.sessions.get(pid)
        if s:
            s.send({"t": "taken", "id": tid}, prio=0)

    def _progress(self, s, tid, direction, frac, done=False):
        job = self.jobs.get(tid)
        last = getattr(job, "_last_frac", -1) if job else -1
        if done or frac - last >= 0.03:
            if job:
                job._last_frac = frac
            self._emit(type="progress", id=tid, peer=s.peer_id, dir=direction, frac=frac, done=done)
        if done:
            self.jobs.pop(tid, None)

    # ---------- incoming messages (session threads) ----------
    def _on_msg(self, s, hdr, payload):
        t = hdr.get("t")
        if not s.trusted:
            self._on_pair_msg(s, t)
            return
        if t == "ghost":
            self._emit(type="ghost", peer=s.peer_id, hands=hdr.get("h", []), holding=bool(hdr.get("hold")))
        elif t == "item":
            self._begin_incoming(s, hdr)
        elif t == "chunk":
            self._chunk(s, hdr, payload)
        elif t == "end":
            self._finish_incoming(s, hdr.get("id"))
        elif t == "taken":
            self._emit(type="taken", peer=s.peer_id, name=s.peer_name, id=hdr.get("id"))
        elif t == "unpaired":
            with self.lock:
                self.ident.pinned.pop(s.peer_id, None)
                self.ident.save()
            self._emit(type="error", message=f"{s.peer_name} unpaired")
            s.close()

    def _on_pair_msg(self, s, t):
        pid = s.peer_id
        emit = None
        with self.lock:
            p = self.pairing.get(pid)
            if t == "pair_req":
                if p and p["state"] == "waiting" and p["session"] is not s:
                    # both asked at once (the "high five"): the lower id's request wins, auto-accepted
                    if self.ident.id < pid:
                        s.close()
                        return
                    p["session"].alive = False
                    try:
                        p["session"].sock.close()
                    except OSError:
                        pass
                    self.pairing[pid] = {"state": "code", "session": s, "sas": s.sas, "me": False,
                                         "them": False, "t": time.time(), "name": s.peer_name}
                    s.send({"t": "pair_ok"}, prio=0)
                    emit = dict(type="pair_code", peer=pid, sas=s.sas)
                elif not p:
                    self.pairing[pid] = {"state": "incoming", "session": s, "sas": s.sas, "me": False,
                                         "them": False, "t": time.time(), "name": s.peer_name}
                    emit = dict(type="pair_request", peer=pid, name=s.peer_name)
            elif p and p["session"] is s:
                if t == "pair_ok" and p["state"] == "waiting":
                    p["state"], p["t"] = "code", time.time()
                    emit = dict(type="pair_code", peer=pid, sas=s.sas)
                elif t == "pair_confirm":
                    p["them"] = True
                elif t == "pair_cancel":
                    del self.pairing[pid]
                    emit = dict(type="pair_failed", peer=pid, reason=f"{s.peer_name} cancelled")
        if emit:
            self._emit(**emit)
        if t == "pair_confirm":
            self._maybe_finish(pid)
        if t == "pair_cancel":
            s.close()

    def _begin_incoming(self, s, hdr):
        tid, kind = str(hdr.get("id")), hdr.get("kind")
        files = hdr.get("files") or []
        total = int(hdr.get("total", 0))
        if kind not in ("text", "image", "files") or total > MAX_ITEM or not files:
            return
        st = {"session": s, "kind": kind, "label": str(hdr.get("label", ""))[:200], "total": total,
              "got": 0, "files": files, "index": -1, "fh": None, "buf": bytearray(), "paths": [], "dir": None}
        if kind == "files":
            stamp = time.strftime("%Y%m%d-%H%M%S")
            safe_name = "".join(c for c in s.peer_name if c.isalnum() or c in " -_")[:30] or "peer"
            st["dir"] = os.path.join(self.recv_dir, f"{stamp} from {safe_name}")
            os.makedirs(st["dir"], exist_ok=True)
        with self.lock:
            self.incoming[tid] = st
        self._emit(type="progress", id=tid, peer=s.peer_id, dir="in", frac=0.0, done=False)

    def _chunk(self, s, hdr, payload):
        tid = str(hdr.get("id"))
        st = self.incoming.get(tid)
        if not st or st["session"] is not s:
            return
        fi = int(hdr.get("f", 0))
        st["got"] += len(payload)
        if st["got"] > st["total"]:
            self._abort_incoming(tid)
            return
        if st["kind"] != "files":
            st["buf"] += payload
        else:
            if fi != st["index"]:
                if st["fh"]:
                    st["fh"].close()
                rel = safe_rel(st["files"][fi][0]) if 0 <= fi < len(st["files"]) else None
                if rel is None:
                    self._abort_incoming(tid)
                    return
                path = os.path.join(st["dir"], rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                st["fh"], st["index"] = open(path, "wb"), fi
            st["fh"].write(payload)
        if st["total"]:
            frac = st["got"] / st["total"]
            if frac - st.get("last", -1) >= 0.03:
                st["last"] = frac
                self._emit(type="progress", id=tid, peer=s.peer_id, dir="in", frac=frac, done=False)

    def _finish_incoming(self, s, tid):
        tid = str(tid)
        with self.lock:
            st = self.incoming.pop(tid, None)
        if not st or st["session"] is not s:
            return
        if st["fh"]:
            st["fh"].close()
        item = {"kind": st["kind"], "label": st["label"], "from": s.peer_name, "peer": s.peer_id, "tid": tid}
        if st["kind"] == "text":
            item["text"] = st["buf"].decode("utf-8", "replace")
        elif st["kind"] == "image":
            item["png"] = bytes(st["buf"])
        else:
            for rel, size in st["files"]:  # zero-byte files never get a chunk
                path = os.path.join(st["dir"], safe_rel(rel) or "_")
                if size == 0 and not os.path.exists(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "wb").close()
            tops = list(dict.fromkeys(f[0].replace("\\", "/").split("/")[0] for f in st["files"]))
            item["paths"] = [os.path.join(st["dir"], t) for t in tops]
        self._emit(type="progress", id=tid, peer=s.peer_id, dir="in", frac=1.0, done=True)
        self._emit(type="item", item=item)

    def _abort_incoming(self, tid):
        st = self.incoming.pop(tid, None)
        if st and st["fh"]:
            st["fh"].close()


# ---------- self-test: two Links in one process over localhost ----------
def selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="airtyper-link-test-")
    results = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
        results.append(bool(ok))

    def wait_for(link, pred, timeout=10.0, keep=None):
        end = time.time() + timeout
        while time.time() < end:
            for e in link.poll():
                if keep is not None:
                    keep.append(e)
                if pred(e):
                    return e
            time.sleep(0.02)
        return None

    def make(tag, udp, other_udp):
        return Link(f"Laptop {tag}", os.path.join(tmp, tag, "link.json"), os.path.join(tmp, tag, "recv"),
                    udp_port=udp, tcp_port=0, manual=[f"127.0.0.1:{other_udp}"])

    a, b = make("A", 47930, 47931), make("B", 47931, 47930)
    a.start(); b.start()
    found = wait_for(a, lambda e: e["type"] == "peers" and b.ident.id in a.peers)
    check("discovery", found)
    a.request_pair(b.ident.id)
    req = wait_for(b, lambda e: e["type"] == "pair_request")
    check("pair request arrives", req and req["name"] == "Laptop A")
    b.accept_pair(a.ident.id)
    ca = wait_for(a, lambda e: e["type"] == "pair_code")
    cb = wait_for(b, lambda e: e["type"] == "pair_code", timeout=2) or {"sas": b.pairing.get(a.ident.id, {}).get("sas")}
    check("both see the same code", ca and cb and ca["sas"] == cb["sas"], ca and ca["sas"])
    a.confirm_pair(b.ident.id); b.confirm_pair(a.ident.id)
    pa = wait_for(a, lambda e: e["type"] == "paired")
    pb = wait_for(b, lambda e: e["type"] == "paired")
    check("paired + pinned", pa and pb and b.ident.id in a.ident.pinned and a.ident.id in b.ident.pinned)

    a.send_item(b.ident.id, {"kind": "text", "text": "hello from A ✓", "label": "hello"})
    got = wait_for(b, lambda e: e["type"] == "item")
    check("text arrives", got and got["item"]["text"] == "hello from A ✓")
    a.taken  # noqa
    b.taken(a.ident.id, got["item"]["tid"])
    check("sender hears it was caught", wait_for(a, lambda e: e["type"] == "taken"))

    src = os.path.join(tmp, "src")
    os.makedirs(os.path.join(src, "folder", "inner"))
    big = os.urandom(CHUNK * 3 + 123)
    with open(os.path.join(src, "big.bin"), "wb") as f:
        f.write(big)
    with open(os.path.join(src, "folder", "inner", "note.txt"), "w") as f:
        f.write("nested")
    open(os.path.join(src, "folder", "empty.txt"), "w").close()
    a.send_item(b.ident.id, {"kind": "files", "paths": [os.path.join(src, "big.bin"), os.path.join(src, "folder")]})
    got = wait_for(b, lambda e: e["type"] == "item", timeout=20)
    ok = False
    if got:
        paths = got["item"]["paths"]
        base = os.path.dirname(paths[0])
        ok = (open(os.path.join(base, "big.bin"), "rb").read() == big
              and open(os.path.join(base, "folder", "inner", "note.txt")).read() == "nested"
              and os.path.exists(os.path.join(base, "folder", "empty.txt"))
              and sorted(os.path.basename(p) for p in paths) == ["big.bin", "folder"])
    check("files + folders arrive intact", ok)

    png = os.urandom(5000)
    a.send_item(b.ident.id, {"kind": "image", "png": png})
    got = wait_for(b, lambda e: e["type"] == "item")
    check("image arrives", got and got["item"]["png"] == png)

    a.send_ghost([[[0.5, 0.5]] * 21], True)
    g = wait_for(b, lambda e: e["type"] == "ghost")
    check("ghost hands arrive", g and g["holding"] and len(g["hands"][0]) == 21)

    # restart B: the pair reconnects by itself
    b.stop()
    wait_for(a, lambda e: e["type"] == "disconnected")
    b2 = make("B", 47931, 47930)
    b2.start()
    rc = wait_for(a, lambda e: e["type"] == "connected", timeout=15)
    check("reconnects after restart", rc)

    # an impostor claiming B's id with a different key is refused
    b2.stop()
    wait_for(a, lambda e: e["type"] == "disconnected")
    os.makedirs(os.path.join(tmp, "X"), exist_ok=True)
    fake = Identity(os.path.join(tmp, "X", "link.json"))
    fake.id = b.ident.id
    fake.save()
    x = Link("Laptop B", os.path.join(tmp, "X", "link.json"), os.path.join(tmp, "X", "recv"),
             udp_port=47931, tcp_port=0, manual=["127.0.0.1:47930"])
    x.start()
    errs = []
    conn = wait_for(a, lambda e: e["type"] == "connected", timeout=10, keep=errs)
    check("impostor refused", not conn and any(e["type"] == "error" for e in errs),
          next((e["message"] for e in errs if e["type"] == "error"), "no error"))
    x.stop(); a.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
