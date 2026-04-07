import json
import urllib.parse
import urllib.request

queries = [
    'random number generator statistical test journal',
    'machine learning randomness evaluation journal',
    'entropy source random bit generator journal',
    'PRNG security analysis journal',
    'NIST SP 800-22 randomness test improvements'
]

for q in queries:
    params = urllib.parse.urlencode({
        'query': q,
        'filter': 'from-pub-date:2021-01-01,type:journal-article',
        'rows': 12,
        'sort': 'relevance'
    })
    url = 'https://api.crossref.org/works?' + params
    print('\n=== QUERY:', q, '===')
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        items = data['message']['items']
        for i, item in enumerate(items[:12], 1):
            title = item.get('title', [''])[0]
            journal = item.get('container-title', [''])[0]
            year = item.get('issued', {}).get('date-parts', [[None]])[0][0]
            doi = item.get('DOI', '')
            authors = item.get('author', [])
            author_names = []
            for a in authors[:4]:
                nm = (a.get('family', '') or a.get('name', '')).strip()
                if nm:
                    author_names.append(nm)
            auth = ', '.join(author_names)
            if len(authors) > 4:
                auth += ', et al.'
            print(f'{i}. {year} | {title} | {journal} | {auth} | DOI:{doi}')
    except Exception as e:
        print('ERROR', e)
