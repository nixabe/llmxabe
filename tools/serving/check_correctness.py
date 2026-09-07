"""Live text/tool acceptance against llama-server on the same GGUF.

Usage: python tools/serving/check_correctness.py --candidate http://127.0.0.1:18080 \
    --reference http://127.0.0.1:18081 --output /tmp/quality.json

Greedy text equality is reported, not assumed: different arithmetic can flip
near ties. Known answers, tool arguments, result replay, streaming, and warm
versus cold/concurrent identity are assertions. This is not a performance bench.
No real tools are executed; the weather result is a fixed fixture.
"""
import argparse
import concurrent.futures
import json
import pathlib
import urllib.request


def request(base, path, body):
    req = urllib.request.Request(base.rstrip('/') + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        if not body.get('stream'):
            return json.load(response)
        events = []
        for line in response:
            if line.startswith(b'data: '):
                data = line[6:].strip()
                if data != b'[DONE]':
                    events.append(json.loads(data))
        return events


def chat_body(prompt, limit=384):
    return {'model': 'test', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': limit,
            'chat_template_kwargs': {'enable_thinking': False}}


def message(result):
    return result['choices'][0]['message']


def streamed_chat(events):
    content, reasoning, calls = '', '', {}
    finish = None
    for event in events:
        for choice in event.get('choices', []):
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta', {})
            content += delta.get('content') or ''
            reasoning += delta.get('reasoning_content') or ''
            for call in delta.get('tool_calls', []):
                item = calls.setdefault(call['index'], {'name': '', 'arguments': ''})
                function = call.get('function', {})
                item['name'] += function.get('name', '')
                item['arguments'] += function.get('arguments', '')
    return content, reasoning, list(calls.values()), finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = {}

    def record(key, value):
        report[key] = value
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    cases = {
        'arithmetic': 'What is 17 times 23? Answer with just the number.',
        'code': 'Write a Python function that removes duplicates from a list while preserving order. Explain its complexity.',
        'unicode': 'Translate “The cat is sleeping” into Traditional Chinese. Give only the translation.',
        'long': 'Records:\n' + ''.join(f'Record {i}: color blue; value {i % 17}.\n' for i in range(650))
                + '\nWhat is the value in Record 617? Answer with just the number.',
    }
    for name, prompt in cases.items():
        pair = [request(base, '/v1/chat/completions', chat_body(prompt))
                for base in [args.candidate, args.reference]]
        exact = message(pair[0]).get('content') == message(pair[1]).get('content')
        record(name, {'exact_text': exact, 'responses': pair})
        assert pair[0]['usage']['prompt_tokens'] == pair[1]['usage']['prompt_tokens'], name
        if name in ('arithmetic', 'long'):
            expected = '391' if name == 'arithmetic' else '5'
            assert all(message(r)['content'].strip() == expected for r in pair), name
        print(f'{name}: exact text={exact}', flush=True)

    # A raw, identical rendered prompt separates model arithmetic from chat
    # preprocessing; it also exercises the completions endpoint.
    prompt = request(args.reference, '/apply-template', chat_body(cases['code']))['prompt']
    raw = [request(base, '/v1/completions', {'model': 'test', 'prompt': prompt,
            'temperature': 0, 'max_tokens': 384}) for base in [args.candidate, args.reference]]
    record('raw', raw)
    assert raw[0]['choices'][0]['text'] == report['code']['responses'][0]['choices'][0]['message']['content']

    # The first long request above was cold in a freshly started candidate.
    expected = message(report['long']['responses'][0])
    for trial in range(2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            prompts = [cases['long'], cases['long'], cases['arithmetic']]
            replies = list(pool.map(lambda p: request(args.candidate, '/v1/chat/completions', chat_body(p)), prompts))
        record(f'cache_batch_{trial}', replies)
        assert message(replies[0]) == message(replies[1]) == expected
        assert message(replies[2])['content'].strip() == '391'

    for temperature in [0.6, 1.0]:
        sampled = dict(chat_body(cases['code']), temperature=temperature,
                       top_k=20, top_p=0.95, min_p=0.05, seed=42)
        replies = [request(args.candidate, '/v1/chat/completions', sampled) for _ in range(2)]
        record(f'seed_replay_{temperature}', replies)
        assert message(replies[0]) == message(replies[1]), 'equal seeds must replay'

    tool = {'type': 'function', 'function': {'name': 'get_weather',
            'description': 'Get current weather for a city.', 'parameters': {
                'type': 'object', 'properties': {'city': {'type': 'string'},
                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
                'required': ['city', 'unit'], 'additionalProperties': False}}}
    body = chat_body('Use get_weather to get the weather in Taipei in celsius.', 256)
    body['tools'] = [tool]
    expected_args = {'city': 'Taipei', 'unit': 'celsius'}
    for label, base in [('candidate', args.candidate), ('reference', args.reference)]:
        reply = request(base, '/v1/chat/completions', body)
        record(f'tool_{label}', reply)
        calls = message(reply)['tool_calls']
        assert reply['choices'][0]['finish_reason'] == 'tool_calls' and len(calls) == 1
        assert calls[0]['function']['name'] == 'get_weather'
        assert json.loads(calls[0]['function']['arguments']) == expected_args
        follow = dict(body, messages=body['messages'] + [message(reply)] + [
            {'role': 'tool', 'tool_call_id': calls[0]['id'],
             'content': '{"temperature":23,"condition":"rain"}'}])
        answer = request(base, '/v1/chat/completions', follow)
        record(f'tool_result_{label}', answer)
        assert '23' in message(answer)['content'] and 'rain' in message(answer)['content'].lower()
        events = request(base, '/v1/chat/completions', dict(body, stream=True))
        record(f'tool_stream_{label}', events)
        _, _, streamed, finish = streamed_chat(events)
        assert finish == 'tool_calls' and len(streamed) == 1
        assert streamed[0]['name'] == 'get_weather'
        assert json.loads(streamed[0]['arguments']) == expected_args
    assert report['tool_candidate']['usage']['prompt_tokens'] == report['tool_reference']['usage']['prompt_tokens']

    typed_tool = {'type': 'function', 'function': {'name': 'submit_report',
                  'description': 'Submit the supplied report.', 'parameters': {
                      'type': 'object', 'properties': {
                          'approved': {'type': 'boolean'}, 'count': {'type': 'integer'},
                          'data': {'type': 'object', 'properties': {
                              'values': {'type': 'array', 'items': {'type': 'integer'}}},
                              'required': ['values']}, 'note': {'type': 'string'}},
                      'required': ['approved', 'count', 'data', 'note']}}}
    typed = dict(chat_body('Call submit_report with approved=true, count=2, '
                          'data={"values":[1,2]}, and note="台北". Do not add prose.', 256),
                 tools=[typed_tool])
    for label, base in [('candidate', args.candidate), ('reference', args.reference)]:
        reply = request(base, '/v1/chat/completions', typed)
        record(f'typed_tool_{label}', reply)
        calls = message(reply)['tool_calls']
        assert len(calls) == 1 and calls[0]['function']['name'] == 'submit_report'
        assert json.loads(calls[0]['function']['arguments']) == {
            'approved': True, 'count': 2, 'data': {'values': [1, 2]}, 'note': '台北'}

    # Check both adapters, including the caller replaying exactly what each
    # endpoint returned. Event protocols differ, so inspect their typed items.
    anthropic = {'model': 'test', 'max_tokens': 256, 'temperature': 0,
                 'thinking': {'type': 'disabled'}, 'messages': body['messages'],
                 'tools': [{'name': 'get_weather', 'description': tool['function']['description'],
                            'input_schema': tool['function']['parameters']}]}
    reply = request(args.candidate, '/v1/messages', anthropic)
    record('anthropic', reply)
    calls = [b for b in reply['content'] if b['type'] == 'tool_use']
    assert len(calls) == 1 and calls[0]['name'] == 'get_weather' and calls[0]['input'] == expected_args
    follow = dict(anthropic, messages=anthropic['messages'] + [
        {'role': 'assistant', 'content': reply['content']},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': calls[0]['id'],
                                     'content': '{"temperature":23,"condition":"rain"}'}]}])
    answer = request(args.candidate, '/v1/messages', follow)
    record('anthropic_result', answer)
    assert any('23' in b.get('text', '') for b in answer['content'])
    events = request(args.candidate, '/v1/messages', dict(anthropic, stream=True))
    record('anthropic_stream', events)
    fragments = ''.join(e.get('delta', {}).get('partial_json', '') for e in events)
    assert json.loads(fragments) == expected_args
    assert any(e.get('delta', {}).get('stop_reason') == 'tool_use' for e in events)

    responses = {'model': 'test', 'max_output_tokens': 256, 'temperature': 0,
                 'reasoning': {'effort': 'none'}, 'input': body['messages'],
                 'tools': [dict(type='function', **tool['function'])]}
    reply = request(args.candidate, '/v1/responses', responses)
    record('responses', reply)
    calls = [item for item in reply['output'] if item['type'] == 'function_call']
    assert len(calls) == 1 and calls[0]['name'] == 'get_weather' and json.loads(calls[0]['arguments']) == expected_args
    follow = dict(responses, input=body['messages'] + reply['output'] + [
        {'type': 'function_call_output', 'call_id': calls[0]['call_id'],
         'output': '{"temperature":23,"condition":"rain"}'}])
    answer = request(args.candidate, '/v1/responses', follow)
    record('responses_result', answer)
    assert '23' in json.dumps(answer['output'])
    events = request(args.candidate, '/v1/responses', dict(responses, stream=True))
    record('responses_stream', events)
    completed = [e['response'] for e in events if e['type'] == 'response.completed']
    assert len(completed) == 1
    calls = [i for i in completed[0]['output'] if i['type'] == 'function_call']
    assert len(calls) == 1 and calls[0]['name'] == 'get_weather' and json.loads(calls[0]['arguments']) == expected_args
    print('PASS: known answers, prompt counts, cache/batch identity, tools and result replay, all three streaming dialects', flush=True)


if __name__ == '__main__':
    main()
