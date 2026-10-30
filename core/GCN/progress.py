from contextlib import contextmanager
import os
import shutil
import sys
import time

def enabled():
    return os.environ.get('QEC_PROGRESS', '1').lower() not in ('0', 'false', 'off', 'no')


def announce(message):
    if enabled():
        print(message, file=sys.stderr, flush=True)


def duration(seconds):
    if seconds is None:
        return '--:--'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


class Progress:
    def __init__(self, total, label, unit='item', stream=None):
        self.total, self.label, self.unit = total, label, unit
        self.stream = sys.stderr if stream is None else stream
        self.active = enabled()
        self.terminal = self.stream.isatty()
        self.interval = .5 if self.terminal else 30.0
        self.completed = 0
        self.started = self.last_display = None
        self.width = 0

    def __enter__(self):
        if self.active:
            self.started = time.monotonic()
            self._display(self.started)
        return self

    def update(self, amount=1):
        self.completed += amount
        if self.active:
            now = time.monotonic()
            if now-self.last_display >= self.interval:
                self._display(now)

    def _display(self, now, status=None):
        elapsed = now-self.started
        remaining = ((self.total-self.completed)*elapsed/self.completed
                     if self.total is not None and self.completed else None)
        if self.total is not None:
            fraction = min(1., self.completed/self.total) if self.total else 1.
            filled = int(10*fraction)
            bar = '[' + '#'*filled + '.'*(10-filled) + ']'
            counts = f'{self.completed:,}/{self.total:,} {self.unit}'
            detail = f'{bar} {int(100*fraction):3d}% {counts} {duration(elapsed)} ETA {duration(remaining)}'
        else:
            detail = f'{self.completed:,} {self.unit} {duration(elapsed)}'
        if status:
            detail += f' {status}'
        label = self.label
        if self.terminal:
            room = max(0, shutil.get_terminal_size((100, 24)).columns-len(detail)-2)
            label = label if len(label) <= room else label[:max(0, room-3)]+'...'
        line = f'{label} {detail}'
        if self.terminal:
            self.stream.write('\r'+line.ljust(self.width)+ ('\n' if status else ''))
            self.width = len(line)
        else:
            self.stream.write(line+'\n')
        self.stream.flush()
        self.last_display = now

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            status = ('done' if self.total is None or self.completed >= self.total else 'stopped') if exc_type is None else 'interrupted'
            self._display(time.monotonic(), status)


def track(iterable, label, total=None, unit='item', count=None):
    if not enabled():
        yield from iterable
        return
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            pass
    with Progress(total, label, unit) as bar:
        for item in iterable:
            yield item
            bar.update(count(item) if count else 1)


@contextmanager
def stage(label):
    if not enabled():
        yield
        return
    start = time.monotonic()
    announce(label)
    try:
        yield
    except BaseException:
        announce(f'{label}: interrupted after {duration(time.monotonic()-start)}')
        raise
    else:
        announce(f'{label}: done in {duration(time.monotonic()-start)}')
