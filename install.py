"""Installer / uninstaller for ornith-serve (Ornith-1.5-35B-A3B on a GTX 1070, Windows).

  python install.py --dir D:\\ornith            install everything into D:\\ornith (resumable: run it again after an abort)
  python install.py --dir D:\\ornith --python .venv\\Scripts\\python.exe     reuse an existing Python environment
  python install.py status                     what is installed where
  python install.py uninstall                  remove what the installer created (asks first; --yes skips the question)

Steps: check GPU / RAM / disk -> Python environment (venv + pinned packages) -> download the checkpoint from Hugging Face
(21.8 GB, resumable) -> split it: non-expert weights + configs into <dir>\\model (5 GB), all routed experts packed into
<dir>\\experts.bin (16.9 GB) -> delete the download -> write install.json + start-ornith.bat -> smoke test.
Peak disk use is about 44 GB in --dir, 22 GB afterwards. Everything is recorded in <dir>\\ornith-install.json, and the
uninstaller removes only what is listed there.
"""
import argparse, ctypes, glob, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = next((d for d in (os.path.join(HERE, 'engine'), os.path.abspath(os.path.join(HERE, '..', 'ornith'))) if os.path.isdir(d)),
              os.path.join(HERE, 'engine'))                    # repository layout: engine/ next to this file
POINTER = os.path.join(HERE, 'install.json')               # the code side: where the data of this installation lives
REPO = 'ornith-ai/Ornith-1.5-35B-A3B-NVFP4'
DOWNLOAD_GB, EXPERTS_GB, MODEL_GB = 21.8, 16.9, 5.0
EXPERT_KEY = re.compile(r'language_model\.layers\.\d+\.mlp\.experts\.\d+\.')   # routed experts -> experts.bin
REQUIREMENTS = os.path.join(HERE, 'requirements.txt')     # pinned versions (PyTorch CUDA 12.6 index inside)
TOOLS = os.path.join(HERE, '.tools')                         # install.bat puts uv + a private Python here if none was found
HOME = os.path.join(os.path.expanduser('~'), '.ornith-serve')


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fail(msg):
    print(f'\nERROR: {msg}', file=sys.stderr, flush=True)
    sys.exit(1)


def free_gb(path):
    if os.environ.get('ORNITH_TEST_FREE_GB'): return float(os.environ['ORNITH_TEST_FREE_GB'])   # tests: pretend a full disk
    while not os.path.exists(path): path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 2 ** 30


def ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [('len', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                    ('tp', ctypes.c_ulonglong), ('ap', ctypes.c_ulonglong), ('tv', ctypes.c_ulonglong), ('av', ctypes.c_ulonglong), ('x', ctypes.c_ulonglong)]
    m = MS(); m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 2 ** 30


class Manifest:
    """<dir>/ornith-install.json: settings, finished steps and every path the installer created"""

    def __init__(self, d):
        self.path = os.path.join(d, 'ornith-install.json')
        self.m = json.load(open(self.path, encoding='utf-8')) if os.path.isfile(self.path) else {'dir': d, 'done': [], 'created': []}

    def save(self):
        with open(self.path + '.tmp', 'w', encoding='utf-8') as f: json.dump(self.m, f, indent=1)
        os.replace(self.path + '.tmp', self.path)

    def done(self, step): return step in self.m['done']

    def mark(self, step):
        if step not in self.m['done']: self.m['done'].append(step)
        self.save()

    def created(self, p):
        p = os.path.abspath(p)
        if p not in self.m['created']: self.m['created'].append(p)
        self.save()


def run(cmd, progress=None, **kw):
    """run a command; progress(proc) -> status line or None, printed every 10 s while it runs"""
    say('>', ' '.join(f'"{c}"' if ' ' in c else c for c in cmd[:3]) + (' ...' if len(cmd) > 3 else ''))
    p = subprocess.Popen(cmd, **kw)
    while True:
        try:
            p.wait(10); break
        except subprocess.TimeoutExpired:
            line = progress(p) if progress else None
            if line: say(line)
    if p.returncode: fail(f'command failed with exit code {p.returncode}: {cmd[0]} ...')


def written_by(p):
    """bytes a process (and its children) wrote so far; None without psutil"""
    try:
        import psutil
        pr = psutil.Process(p.pid)
        return sum(x.io_counters().write_bytes for x in [pr] + pr.children(recursive=True))
    except Exception:
        return None


def rate_line(what, total_gb):
    """progress callback: '<what> 12.3 / 21.8 GB, 38 MB/s, about 4 min left' from the bytes the process wrote"""
    st = {'t0': time.time(), 'hist': []}

    def f(p):
        b = written_by(p)
        if b is None: return None
        now = time.time()
        st['hist'] = [(t, x) for t, x in st['hist'] if now - t <= 60] + [(now, b)]   # rate over the last minute: Windows
        t1, b1 = st['hist'][0] if len(st['hist']) > 1 else (st['t0'], 0)              # writes in bursts, 10 s alone jumps 0..max
        v = (b - b1) / max(1e-6, now - t1)
        left = (total_gb * 2 ** 30 - b) / v / 60 if v > 1e5 else None
        eta = '' if not left or left <= 0 else ', less than a minute left' if left < 1 else f', about {left:.0f} min left'
        return f'{what} {b / 2 ** 30:.1f} / {total_gb} GB, {v / 2 ** 20:.0f} MB/s' + eta
    return f


def slow_disk(path):
    """True if path is on a USB / spinning disk (Windows)"""
    if os.name != 'nt': return False
    try:
        d = os.path.splitdrive(os.path.abspath(path))[0].rstrip(':')
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              f'$d = Get-Partition -DriveLetter {d} | Get-Disk; "$($d.BusType)"; (Get-PhysicalDisk | Where-Object DeviceId -eq $d.Number).MediaType'],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return any(x in ('USB', 'HDD') for x in out)
    except Exception:
        return False


# ====================================================================================================== steps
def check(a):
    say('checking the PC')
    if os.name != 'nt': say('warning: only tested on Windows 10/11')
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)) and not a.python:
        fail(f'Python {sys.version.split()[0]}: use Python 3.10 to 3.12 (the pinned packages exist for these)')
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        name, mem, cc, drv = [x.strip() for x in q.split(',')]
    except Exception:
        fail('nvidia-smi not found: install the NVIDIA driver first')
    say(f'GPU {name}, {mem} MiB, sm_{cc.replace(".", "")}, driver {drv}')
    if (cc != '6.1' or int(mem) < 7600) and not a.force:
        fail(f'{name} is not a GTX 1070-class card (Pascal sm_61, 8 GB). ornith-serve is built for it only; --force installs anyway.')
    r = ram_gb() if os.name == 'nt' else 0
    if r and r < 15: fail(f'{r:.0f} GB RAM: at least 16 GB are needed (32 GB recommended)')
    if r and r < 30: say(f'{r:.0f} GB RAM: start the server with --ram-gb {max(1, int(r - 12))} (fewer experts in RAM, slower)')
    need = 0 if a.weights_from else MODEL_GB + EXPERTS_GB + (DOWNLOAD_GB if not os.path.isfile(os.path.join(a.dir, 'experts.bin')) else 0)
    if not a.python: need += 7                                    # venv with PyTorch + CUDA libraries
    have = free_gb(a.dir)
    say(f'disk {a.dir}: {have:.0f} GB free, about {need:.0f} GB needed during the install, {MODEL_GB + EXPERTS_GB:.0f} GB afterwards')
    if have < need + 1 and not a.force:
        fail(f'not enough space in {a.dir}: {need + 1 - have:.0f} GB missing. Free some space or choose another --dir '
             f'(an external drive works, just slower).')
    if slow_disk(a.dir):
        say(f'warning: {a.dir} is on a USB or spinning disk. The install will take much longer (the download is written in '
            f'pieces) and the server starts slower. An internal SSD is strongly recommended.')


def python_env(a, man):
    if a.python:
        py = os.path.abspath(a.python)
        if not os.path.isfile(py): fail(f'--python {py} does not exist')
        man.m.update(python=py, venv_created=False); man.save()
    elif not man.done('venv'):
        venv = os.path.join(a.dir, 'venv')
        say(f'creating the Python environment in {venv} (about 6 GB with CUDA libraries)')
        run([sys.executable, '-m', 'venv', venv])
        man.created(venv)
        py = os.path.join(venv, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv, 'bin', 'python')
        run([py, '-m', 'pip', 'install', '--upgrade', 'pip'])
        say('installing the packages (PyTorch with CUDA is a 2.5 GB download)')
        run([py, '-m', 'pip', 'install', '-r', REQUIREMENTS], progress=rate_line('packages', 6.0))
        man.m.update(python=py, venv_created=True); man.mark('venv')
    py = man.m['python']
    run([py, '-c', 'import torch, cupy, transformers, safetensors, huggingface_hub; assert torch.cuda.is_available(), "torch sees no CUDA GPU"; '
                   'print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))'])
    return py


def download(a, man, py):
    dl = os.path.join(a.dir, 'download')
    if man.done('download') or man.done('pack'): return dl
    say(f'downloading {REPO} ({DOWNLOAD_GB} GB) to {dl}; an abort can be resumed by running the installer again')
    man.created(dl)
    code = (f'from huggingface_hub import snapshot_download\n'
            f'snapshot_download({REPO!r}, local_dir={dl!r}, max_workers=4, ignore_patterns=["assets/*", "*.png"])\n')
    run([py, '-c', code], progress=rate_line('download', DOWNLOAD_GB),
        # HF_HUB_DISABLE_XET: the Xet backend writes files in many pieces at random places; on a hard disk that falls to
        # ~10 MB/s. The plain HTTP download writes each file in one go.
        env=dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING='1', HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_XET='1'))
    shards = glob.glob(os.path.join(dl, 'model-*.safetensors'))
    if len(shards) != 3: fail(f'download incomplete: {len(shards)} of 3 weight files in {dl}')
    man.mark('download')
    return dl


def reuse_weights(a, man):
    """--weights-from DIR: use the model + experts.bin of another installation in place (they are not copied and the
    uninstaller of this installation leaves them alone)"""
    src = os.path.abspath(a.weights_from)
    model, experts = os.path.join(src, 'model'), os.path.join(src, 'experts.bin')
    if not os.path.isdir(model) or not os.path.isfile(experts) or os.path.getsize(experts) < EXPERTS_GB * 0.95 * 2 ** 30:
        fail(f'--weights-from {src}: no complete model\\ + experts.bin there')
    say(f'using the weights in {src}')
    man.m['weights_from'] = src; man.mark('model'); man.mark('pack')
    return model, experts


def split(a, man, py, dl):
    model = os.path.join(a.dir, 'model')
    if not man.done('model'):
        say(f'writing configs + non-expert weights to {model}')
        man.created(model)
        run([py, os.path.abspath(__file__), '_slim', dl, model], progress=rate_line('non-expert weights', MODEL_GB))
        man.mark('model')
    experts = os.path.join(a.dir, 'experts.bin')
    if not man.done('pack'):
        say(f'packing the experts into {experts} (40 layers; on a USB disk this takes a while)')
        man.created(experts)
        run([py, os.path.join(ENGINE, 'pack.py')], progress=rate_line('experts', EXPERTS_GB), cwd=ENGINE,
            env=dict(os.environ, ORNITH_MODEL=dl, ORNITH_EXPERTS=experts))
        if not os.path.isfile(experts) or os.path.getsize(experts) < EXPERTS_GB * 0.95 * 2 ** 30:
            fail(f'{experts} is incomplete: run the installer again')
        man.mark('pack')
    if not a.keep_download and os.path.isdir(dl):
        say(f'deleting the download {dl} (no longer needed)')
        shutil.rmtree(dl, ignore_errors=True)
    return model, experts


def slim(src, dst):
    """(runs inside the target Python) configs + tokenizer + every tensor that is not a routed expert -> dst"""
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(('.json', '.jinja', '.txt', '.md')) and os.path.isfile(os.path.join(src, f)) and f != 'model.safetensors.index.json':
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    shards = sorted(glob.glob(os.path.join(src, 'model-*.safetensors')))
    n = 0
    for i, p in enumerate(shards):
        out = os.path.join(dst, f'nonexpert-{i + 1:05d}-of-{len(shards):05d}.safetensors')
        if os.path.isfile(out): continue
        with safe_open(p, framework='pt') as h:
            t = {k: h.get_tensor(k) for k in h.keys() if not EXPERT_KEY.search(k)}
        save_file(t, out + '.tmp', metadata={'format': 'pt'}); os.replace(out + '.tmp', out)
        n += len(t)
        say(f'{os.path.basename(out)}: {len(t)} tensors')
    say(f'non-expert weights done ({n} tensors written this run)')


def finish(a, man, py, model, experts):
    pointer = {'dir': a.dir, 'python': py, 'model_dir': model, 'experts_bin': experts, 'engine_dir': ENGINE, 'installed': time.strftime('%Y-%m-%d %H:%M')}
    with open(POINTER, 'w', encoding='utf-8') as f: json.dump(pointer, f, indent=1)
    man.m['pointer'] = POINTER
    bat = os.path.join(HERE, 'start-ornith.bat')
    with open(bat, 'w', encoding='utf-8') as f:
        f.write('@echo off\r\nrem ornith-serve: OpenAI API at http://127.0.0.1:8000/v1, Anthropic API at http://127.0.0.1:8000\r\n'
                f'"{py}" "{os.path.join(HERE, "ornith_serve.py")}" %*\r\npause\r\n')
    man.created(bat); man.created(POINTER)
    man.mark('installed')
    say(f'start the server with: {bat}   (options: python ornith_serve.py --help)')


def smoke(man):
    """start the server, ask one question over the OpenAI API, stop it"""
    import urllib.request
    py, port = man.m['python'], 18999
    say('smoke test: starting the server (the first start loads ~17 GB of experts; on a USB disk give it a few minutes)')
    p = subprocess.Popen([py, os.path.join(HERE, 'ornith_serve.py'), '--port', str(port), '--snapshots', ''], cwd=HERE)
    try:
        t0 = time.time()
        while time.time() - t0 < 1800:
            if p.poll() is not None: fail('the server stopped during start-up (see the messages above)')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as r:
                    if json.loads(r.read())['status'] == 'ready': break
            except Exception:
                pass
            time.sleep(5)
        else:
            fail('the server was not ready after 30 minutes')
        body = json.dumps({'messages': [{'role': 'user', 'content': 'Antworte nur mit einem Wort: Wie heißt die Hauptstadt von Frankreich?'}],
                           'reasoning_effort': 'none', 'max_tokens': 20, 'temperature': 0}).encode()
        t1 = time.time()
        with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'}), timeout=600) as r:
            ans = json.loads(r.read())
        txt = ans['choices'][0]['message']['content']
        say(f'smoke test answer: {txt!r} ({time.time() - t1:.1f}s, start-up {t1 - t0:.0f}s)')
        if 'paris' not in txt.lower(): say('warning: unexpected answer - check the install')
    finally:
        p.terminate()
        try: p.wait(30)
        except subprocess.TimeoutExpired: p.kill()


# ====================================================================================================== status / uninstall
def find_manifest(a):
    d = a.dir
    if not d and os.path.isfile(POINTER):
        d = json.load(open(POINTER, encoding='utf-8')).get('dir')
    if not d or not os.path.isfile(os.path.join(d, 'ornith-install.json')):
        fail('no installation found (give --dir of the installation)')
    return Manifest(d)


def size_of(p):
    if os.path.isfile(p): return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def status(a):
    man = find_manifest(a)
    say(f'installation in {man.m["dir"]}: steps done {man.m["done"]}')
    for p in man.m['created']:
        say(f'  {p}: ' + (f'{size_of(p) / 2 ** 30:.2f} GB' if os.path.exists(p) else 'missing'))


def uninstall(a):
    man = find_manifest(a)
    items = [q for p in man.m['created'] for q in (p, p + '.tmp') if os.path.exists(q)]   # .tmp: a step that was aborted
    extra = [p for p in (HOME,) if os.path.exists(p) and not a.keep_settings]   # settings, hot experts, conversation snapshots
    print('\nThe uninstaller deletes:')
    for p in items + extra: print(f'  {p}  ({size_of(p) / 2 ** 30:.2f} GB)')
    print(f'  {man.path}')
    print('The program code (this folder) stays.')
    if not a.yes and input('\nDelete these files? Type "yes": ').strip().lower() not in ('yes', 'ja', 'y', 'j'):
        print('nothing deleted'); return
    later = []
    for p in items + extra:
        if os.path.abspath(sys.executable).startswith(os.path.abspath(p) + os.sep) and os.name == 'nt':
            # the uninstaller runs on the private Python inside this folder: delete it right after this process ends
            subprocess.Popen(f'cmd /c ping -n 4 127.0.0.1 >nul & rmdir /s /q "{p}"', creationflags=0x00000008 | 0x00000200,   # string: keep the quotes as they are
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            say(f'{p} is deleted a few seconds after the uninstaller ends')
            later.append(os.path.abspath(p))
            continue
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            say(f'deleted {p}')
        except OSError as ex:
            say(f'could not delete {p}: {ex} (is the server still running?)')
    os.remove(man.path)
    d = man.m['dir']
    rest = [f for f in os.listdir(d) if os.path.join(d, f) not in later] if os.path.isdir(d) else []
    if os.path.isdir(d) and not os.listdir(d):
        try: os.rmdir(d)
        except OSError: pass
    elif rest:
        say(f'{d} still contains files the installer did not create: left in place')
    say('uninstalled')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_slim': return slim(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser(description='install / uninstall ornith-serve with the Ornith-1.5-35B-A3B weights',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('action', nargs='?', default='install', choices=['install', 'status', 'uninstall'])
    ap.add_argument('--dir', help='where the weights (and the Python environment) go, about 22 GB (44 GB during the install)')
    ap.add_argument('--python', help='use this existing Python (with the packages) instead of creating a venv')
    ap.add_argument('--weights-from', help='use the weights of another installation folder instead of downloading them')
    ap.add_argument('--keep-download', action='store_true', help='keep the original Hugging Face files (21.8 GB more)')
    ap.add_argument('--no-test', action='store_true', help='skip the smoke test at the end')
    ap.add_argument('--force', action='store_true', help='install although the GPU / disk checks fail')
    ap.add_argument('--yes', action='store_true', help='uninstall without asking')
    ap.add_argument('--keep-settings', action='store_true', help=f'uninstall: keep {HOME} (settings, conversation snapshots)')
    a = ap.parse_args()
    if a.dir: a.dir = os.path.abspath(a.dir)
    if a.action == 'status': return status(a)
    if a.action == 'uninstall': return uninstall(a)
    if not a.dir: fail('give the target folder: python install.py --dir D:\\ornith')
    check(a)                                                      # before anything is created
    os.makedirs(a.dir, exist_ok=True)
    man = Manifest(a.dir)
    if os.path.abspath(sys.executable).startswith(TOOLS + os.sep): man.created(TOOLS)   # private Python from install.bat
    py = python_env(a, man)
    if a.weights_from:
        model, experts = reuse_weights(a, man)
    else:
        dl = download(a, man, py)
        model, experts = split(a, man, py, dl)
    finish(a, man, py, model, experts)
    if not a.no_test: smoke(man)
    say('installation complete')


if __name__ == '__main__':
    main()
