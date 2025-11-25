# notify_ntfy.py
import os, sys, time, traceback, socket, getpass, platform, runpy, argparse, shlex, json
import urllib.request, urllib.error

# ============ 유틸 ============
def _duration(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _default_title():
    return os.getenv("JOB_TITLE") or os.path.basename(sys.argv[0]) or "Python Job"

def _git_info():
    try:
        import subprocess
        c = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        b = subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return c, b
    except Exception:
        return None, None

def _env(name, default=None):
    v = os.getenv(name)
    return v if v is not None and v != "" else default

# ============ ntfy 전송 ============
import urllib.request, urllib.error, time

def _latin1_safe(s):
    if s is None:
        return None
    # 헤더는 latin-1만 허용 → 이모지/한글은 드롭
    return s.encode("latin-1", "ignore").decode("latin-1")

def _post_ntfy(server, topic, token, subject, body, tags=None, priority=None, markdown=True, retries=3, timeout=15):
    server = (server or "https://ntfy.sh").rstrip("/")
    topic = topic or "my-python-jobs"
    url = f"{server}/{topic}"

    headers = {
        "Title": _latin1_safe(subject),                     # ← 안전화
        "Content-Type": "text/plain; charset=utf-8"
    }
    if markdown:
        headers["Markdown"] = "yes"
    if tags:
        headers["Tags"] = _latin1_safe(tags)               # ← 안전화(혹시 비ASCII 들어오면 제거)
    if priority:
        headers["Priority"] = str(priority)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return
        except Exception as e:
            last_err = e
            wait = 2 ** i
            print(f"[notify_ntfy] 전송 실패(try {i+1}/{retries}): {e} → {wait}s 후 재시도")
            time.sleep(wait)
    raise RuntimeError(f"[notify_ntfy] ntfy 전송 실패: {last_err}")
# --- 교체 끝 ---


def _send_ntfy(subject, body, tags=None, priority=None, server=None, topic=None, token=None):
    server = server or _env("NTFY_SERVER", "https://ntfy.sh")
    topic  = topic  or _env("NTFY_TOPIC", "my-python-jobs")
    token  = token  or _env("NTFY_TOKEN", "")
    _post_ntfy(server, topic, token, subject, body, tags=tags, priority=priority, markdown=True)

# ============ 데코레이터 ============
def ntfy_notify(title=None, notify_on="both", tags_start="arrow_forward", tags_ok="white_check_mark", tags_fail="x", priority=None):
    """
    notify_on: "both" | "success" | "fail" | "start"
    tags_*: ntfy 이모지 태그(쉼표로 여러 개 가능) ex) "white_check_mark,rocket"
    priority: 1..5 (ntfy 우선순위)
    """
    def deco(fn):
        def wrapper(*args, **kwargs):
            job_title = title or _default_title()
            host = socket.gethostname()
            user = getpass.getuser()
            py = platform.python_version()
            commit, branch = _git_info()
            argv = " ".join(shlex.quote(a) for a in sys.argv)
            cwd = os.getcwd()

            start_ts = time.time()
            if notify_on in ("both","start"):
                _send_ntfy(f"▶️ {job_title} 시작",
                           f"argv: `{argv}`\n"
                           f"host: `{host}` | user: `{user}` | py: `{py}`\n"
                           f"cwd : `{cwd}`\n"
                           f"git : `{branch or '-'}` @ `{commit or '-'}`\n"
                           f"시작: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                           tags=tags_start, priority=priority)

            try:
                out = fn(*args, **kwargs)
                took = _duration(time.time() - start_ts)
                if notify_on in ("both","success"):
                    _send_ntfy(f"✅ {job_title} 완료 ({took})",
                               f"소요시간: *{took}*\n종료: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                               tags=tags_ok, priority=priority)
                return out
            except Exception:
                took = _duration(time.time() - start_ts)
                tb = "".join(traceback.format_exc())
                if len(tb) > 4000:
                    tb = tb[-4000:]
                if notify_on in ("both","fail"):
                    _send_ntfy(f"❌ {job_title} 실패 ({took})",
                               f"소요시간: *{took}*\n에러 트레이스:\n```{tb}```",
                               tags=tags_fail, priority=priority)
                raise
        return wrapper
    return deco

# ============ 래퍼 모드 ============
def _run_module_with_argv(module, rest):
    sys.argv = [module] + rest
    return runpy.run_module(module, run_name="__main__")

def _main():
    p = argparse.ArgumentParser(description="ntfy 푸시 알림 래퍼")
    p.add_argument("-m","--module", help="실행할 모듈 (예: pkg.train)")
    p.add_argument("--title", default=None, help="알림 제목")
    p.add_argument("--notify-on", default="both", choices=["both","success","fail","start"])
    p.add_argument("--server", default=_env("NTFY_SERVER","https://ntfy.sh"))
    p.add_argument("--topic",  default=_env("NTFY_TOPIC","my-python-jobs"))
    p.add_argument("--token",  default=_env("NTFY_TOKEN",""))
    p.add_argument("--tags-start", default="arrow_forward")
    p.add_argument("--tags-ok",    default="white_check_mark")
    p.add_argument("--tags-fail",  default="x")
    p.add_argument("--priority",   type=int, default=None)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="-- 이후는 모듈 인자로 전달")
    args = p.parse_args()

    if not args.module:
        print("예) python -m notify_ntfy -m your.module -- --epochs 10")
        sys.exit(2)

    job_title = args.title or f"python -m {args.module}"
    start_ts = time.time()

    if args.notify_on in ("both","start"):
        _post_ntfy(args.server, args.topic, args.token,
                   f"▶️ {job_title} 시작",
                   f"argv: `{' '.join(args.rest)}`\n시작: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                   tags=args.tags_start, priority=args.priority)

    try:
        _run_module_with_argv(args.module, args.rest)
        took = _duration(time.time() - start_ts)
        if args.notify_on in ("both","success"):
            _post_ntfy(args.server, args.topic, args.token,
                       f"✅ {job_title} 완료 ({took})",
                       f"소요시간: *{took}*",
                       tags=args.tags_ok, priority=args.priority)
    except SystemExit as e:
        code = int(getattr(e, "code", 0) or 0)
        took = _duration(time.time() - start_ts)
        if code == 0:
            if args.notify_on in ("both","success"):
                _post_ntfy(args.server, args.topic, args.token,
                           f"✅ {job_title} 종료 코드 0 ({took})",
                           "정상 종료",
                           tags=args.tags_ok, priority=args.priority)
        else:
            if args.notify_on in ("both","fail"):
                _post_ntfy(args.server, args.topic, args.token,
                           f"❌ {job_title} 종료 코드 {code} ({took})",
                           "비정상 종료",
                           tags=args.tags_fail, priority=args.priority)
        raise
    except Exception:
        took = _duration(time.time() - start_ts)
        tb = "".join(traceback.format_exc())
        if len(tb) > 4000:
            tb = tb[-4000:]
        if args.notify_on in ("both","fail"):
            _post_ntfy(args.server, args.topic, args.token,
                       f"❌ {job_title} 실패 ({took})",
                       f"에러:\n```{tb}```",
                       tags=args.tags_fail, priority=args.priority)
        raise

if __name__ == "__main__":
    _main()
