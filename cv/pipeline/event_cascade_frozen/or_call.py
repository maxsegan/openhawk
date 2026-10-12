"""OpenRouter runner for the vlm_event_benchmark labeller axis.

Sends the SAME prompt over the SAME storyboard images as the codex/astra arms, four panels per
call, and writes the raw body to bNNN.json so scripts/score_labeller.py's parser can read it.
"""
import base64, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

def key():
    # Public release: the key comes from the environment only (paid rollback path).
    value = os.environ.get('OPENROUTER_API_KEY')
    if not value:
        raise SystemExit('no OPENROUTER_API_KEY')
    return value

KEY = os.environ.get('OPENROUTER_API_KEY', '')
URL = 'https://openrouter.ai/api/v1/chat/completions'


def call(model, prompt, images, max_tokens=8000, timeout=900, extra=None, base_url=None):
    """base_url points the same request at a local OpenAI-compatible server (vLLM) instead."""
    content = [{'type': 'text', 'text': prompt}]
    for p in images:
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens, 'temperature': 0.0, 'usage': {'include': True}}
    if extra:
        body.update(extra)
    url = base_url.rstrip('/') + '/chat/completions' if base_url else URL
    if base_url:
        body.pop('usage', None)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Authorization': f'Bearer {KEY}',
                                          'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


if __name__ == '__main__':
    model = sys.argv[1]
    imgs = sys.argv[2:]
    prompt = Path('data/processed/vlm_event_benchmark_v1/scripts/mint_refine_prompt.txt').read_text()
    ids = ' '.join(Path(i).stem for i in imgs)
    t0 = time.time()
    out = call(model, prompt + f'\n\nThe attached storyboards are, in order, ids: {ids}', imgs)
    print('dt', round(time.time() - t0, 1))
    print(json.dumps(out.get('usage'), indent=1))
    print(out['choices'][0]['message'].get('content', '')[:3000])
