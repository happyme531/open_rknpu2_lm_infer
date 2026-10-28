#!/usr/bin/env python3
"""Drive the actual terminal UI with real photos and inspect rendered screens."""
import json
import re
import sys
import time

from vision_support import arguments, command, environment


def main():
    args = arguments(__doc__, terminal=True)
    if args.python_deps:
        sys.path.insert(0, str(args.python_deps.resolve()))
    import pexpect
    import pyte

    cmd = command(args) + ['chat', '--temp', '0', '--n-predict', '64']
    report = {'command': cmd, 'cases': [], 'passed': False}
    raw = (args.output/'terminal.ansi').open('w', encoding='utf-8', newline='')
    child = pexpect.spawn(cmd[0], cmd[1:], env=environment('warn'), encoding='utf-8',
                          codec_errors='replace', dimensions=(48, 160), timeout=300)
    # Crossterm raw mode treats LF as Ctrl+J; the Enter key sends CR.
    child.linesep = '\r'
    child.logfile_read = raw

    class Screen(pyte.Screen):
        def write_process_input(self, data):
            child.send(data)

    screen = Screen(160, 48)
    stream = pyte.Stream(screen)

    def rendered():
        return '\n'.join(screen.display)

    def wait(predicate, timeout=240):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            try:
                data = child.read_nonblocking(65536, timeout=.2)
                stream.feed(data)
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                raise RuntimeError('TUI exited unexpectedly: '+rendered())
            view = rendered()
            if 'Error:' in screen.display[-1]:
                raise RuntimeError(view)
            if predicate(view):
                return view
        raise TimeoutError(rendered())

    def snapshot(name):
        (args.output/(name+'.txt')).write_text('\n'.join(line.rstrip() for line in screen.display)+'\n')

    def attach(name):
        child.sendline('/image '+str(args.photos/name))
        wait(lambda view: '1 image(s) attached to next message' in view)
        snapshot('attach-'+name)

    def message(name, prompt, checks):
        child.sendline(prompt)
        wait(lambda view: 'Generating' in screen.display[-1])
        wait(lambda view: re.search(r'\d+ prompt.*\d+ output.*(?:stop|length)', screen.display[-1]))
        view = rendered()
        snapshot(name)
        body = view.rsplit('Assistant', 1)[-1].split(' Message ', 1)[0]
        answer = ' '.join(line.strip(' │┌┐└┘─') for line in body.splitlines()).strip()
        footer = screen.display[-1].strip()
        usage = re.search(r'(\d+) prompt.*?(\d+) cached.*?(\d+) output', footer)
        assert usage, footer
        row = {'name': name, 'prompt': prompt, 'answer': answer, 'footer': footer,
               'prompt_tokens': int(usage[1]), 'cached_tokens': int(usage[2]),
               'completion_tokens': int(usage[3]),
               'semantic_checks': [{'any_of': words, 'passed': any(word in answer.lower() for word in words)} for words in checks]}
        report['cases'].append(row)
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        print(name, answer, flush=True)
        return row

    try:
        wait(lambda view: 'Ready' in screen.display[-1] and 'Send a message to begin.' in view, timeout=300)
        snapshot('ready')
        attach('cats.png')
        first = message('cats', 'How many cats are in this photograph? Answer briefly.', [('two', '2', 'pair')])
        followup = message('cats-followup', 'What color is the surface they are lying on?', [('pink', 'red')])
        assert followup['prompt_tokens'] > first['prompt_tokens']
        attach('bus.jpg')
        message('bus', 'Now describe the large vehicle in this new photograph briefly.', [('bus', 'coach')])
        message('bus-followup', 'What colors is that vehicle?', [('blue',)])
        child.sendline('/reset')
        wait(lambda view: 'Ready' in screen.display[-1] and 'Send a message to begin.' in view)
        snapshot('reset')
        last = message('text-after-reset', '请只回复 OK。', [('ok',)])
        assert last['prompt_tokens'] < first['prompt_tokens']
        report['passed'] = all(check['passed'] for row in report['cases'] for check in row['semantic_checks'])
        child.sendcontrol('c')
        child.expect(pexpect.EOF, timeout=30)
        child.close()
        report['exit_code'] = child.exitstatus
        assert child.exitstatus == 0, child.exitstatus
        assert report['passed'], 'review terminal screenshots and answers'
    finally:
        if child.isalive():
            child.terminate(force=True)
        raw.close()
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
