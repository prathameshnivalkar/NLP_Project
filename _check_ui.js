// Headless smoke test for the Exp06-10 renderers.
// Stubs a minimal DOM so the real renderers run without a browser.
const fs = require('fs');

const nodes = new Map();
function makeNode(id) {
  return {
    id,
    innerHTML: '',
    textContent: '',
    hidden: false,
    dataset: {},
    setAttribute() {},
    getAttribute: () => null,
    classList: { add() {}, remove() {}, contains: () => false },
    style: {},
    querySelector: () => null,
  };
}
global.document = {
  querySelector(selector) {
    const id = selector.replace('#', '');
    if (!nodes.has(id)) nodes.set(id, makeNode(id));
    return nodes.get(id);
  },
  querySelectorAll() { return []; },
  addEventListener() {},
};
global.window = { setTimeout, clearTimeout, addEventListener() {} };
global.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 0);

const source = fs.readFileSync('static/js/app.js', 'utf8');
// Expose the renderers by evaluating the file, then grabbing the functions.
const sandbox = new Function(`${source}\n;return { renderPos, renderChunks, renderEntities, renderFeatureStudy, renderSimilarity, renderWsd, renderDashboard, fallbackData };`);
const api = sandbox();

function expect(cond, label) {
  if (!cond) { console.error(`  FAIL ${label}`); process.exitCode = 1; }
  else console.log(`  ok   ${label}`);
}

const populated = {
  pos: {
    tokens: 100, sentences: 4, baseline: 'hybrid',
    distribution: [{ group: 'Noun', count: 40, share: 40 }],
    tagger_comparison: [{ tagger: 'perceptron', agreement: 100 }, { tagger: 'rules', agreement: 55.3 }, { tagger: 'hybrid', agreement: 98.7 }],
    key_adjectives: ['clear'], key_nouns: ['audio'], key_verbs: ['explains'],
    examples: { Noun: [{ word: 'audio', tag: 'NN', group: 'Noun' }] },
  },
  chunks: {
    total_chunks: 9, avg_per_sentence: 2.2,
    types: [{ type: 'NP', label: 'Noun Phrase', count: 6, share: 66.7 }],
    example: { text: 'x', chunks: [{ label: 'Noun Phrase', text: 'the audio' }] },
  },
  entities: {
    total: 3,
    types: [{ type: 'Technology', count: 2, share: 66.7 }],
    top_entities: [{ text: 'Python', type: 'Technology', count: 2, share: 66.7 }],
    by_source: {}, examples: {},
  },
  feature_study: {
    documents: 20, positive: 10, negative: 10, duplicates_removed: 2,
    feature_sets: [{ key: 'unigrams', label: 'Words only' }, { key: 'all', label: 'Words + POS + chunks' }],
    fractions: [0.5, 1.0],
    rows: [
      { feature_set: 'unigrams', label: 'Words only', fraction: 0.5, train_size: 10, accuracy: 60, f1: 55, vocabulary: 20 },
      { feature_set: 'unigrams', label: 'Words only', fraction: 1.0, train_size: 20, accuracy: 70, f1: 68, vocabulary: 30 },
      { feature_set: 'all', label: 'Words + POS + chunks', fraction: 0.5, train_size: 10, accuracy: 80, f1: 78, vocabulary: 40 },
      { feature_set: 'all', label: 'Words + POS + chunks', fraction: 1.0, train_size: 20, accuracy: 90, f1: 89, vocabulary: 50 },
    ],
    best: { feature_set: 'all', label: 'Words + POS + chunks', fraction: 1.0, train_size: 20, accuracy: 90, f1: 89, vocabulary: 50 },
    reason: '',
  },
  similarity: {
    comments: 10, groups: 2, grouped_comments: 6, redundant: 4, compression: 40.0,
    redundant_pairs: 5, threshold: 0.3, mixed_groups: 0,
    rows: [
      { label: 'Audio Quality', size: 4, share: 40, sentiment: 'negative', cohesion: 0.7, example: 'poor audio' },
      { label: 'Tutorial Thank', size: 2, share: 20, sentiment: 'positive', cohesion: 0.6, example: 'great tutorial' },
    ],
    redundancy_note: 'note',
  },
  wsd: {
    words: 1, disambiguated: 4, ambiguous_words: ['bank'],
    model: { engine: 'GRU', words: 48, examples: 795, parameters: 80080, train_accuracy: 99.1, final_loss: 0.028, trained: true, cached: true },
    rows: [{
      word: 'bank', occurrences: 4, senses_used: 2,
      top_sense: 'depository_financial_institution.n.01',
      top_gloss: 'a financial institution that accepts deposits',
      agreement: 0.5,
      breakdown: [{ sense: 'depository_financial_institution.n.01', count: 2, share: 0.5 }, { sense: 'bank.n.01', count: 2, share: 0.5 }],
    }],
    examples: [],
  },
};

console.log('== populated payload');
api.renderPos(populated);
expect(nodes.get('pos-tokens').textContent.includes('100'), 'pos token count set');
expect(nodes.get('pos-compare').innerHTML.includes('rules'), 'tagger comparison rendered');
expect(nodes.get('pos-compare').innerHTML.includes('(baseline)'), 'baseline labelled');
expect(nodes.get('pos-sentence').innerHTML.includes('audio'), 'tagged example rendered');
api.renderChunks(populated);
expect(nodes.get('chunk-types').innerHTML.includes('Noun Phrase'), 'chunk types rendered');
expect(nodes.get('chunk-example').innerHTML.includes('the audio'), 'chunk example rendered');
api.renderEntities(populated);
expect(nodes.get('entity-chips').innerHTML.includes('Python'), 'entity chips rendered');
expect(nodes.get('entity-count').textContent.includes('3'), 'entity count set');
api.renderFeatureStudy(populated);
expect(nodes.get('study-chart').innerHTML.includes('Words only'), 'study series rendered');
expect(nodes.get('study-insight').innerHTML.includes('85.7') || nodes.get('study-insight').innerHTML.includes('89.0'), 'study insight cites best F1');
api.renderSimilarity(populated);
expect(nodes.get('similarity-rows').innerHTML.includes('Audio Quality'), 'similarity rows rendered');
expect(nodes.get('similarity-gain').textContent.includes('40'), 'compression shown');
api.renderWsd(populated);
expect(nodes.get('wsd-rows').innerHTML.includes('bank'), 'wsd rows rendered');
expect(nodes.get('wsd-engine').textContent.includes('GRU'), 'wsd engine label');
expect(nodes.get('wsd-note').innerHTML.includes('senses'), 'wsd note explains senses');

console.log('\n== empty / degenerate payload');
const empty = {
  pos: { tokens: 0, sentences: 0, baseline: 'perceptron', distribution: [], tagger_comparison: [], key_adjectives: [], key_nouns: [], key_verbs: [], examples: {} },
  chunks: { total_chunks: 0, avg_per_sentence: 0, types: [], example: { text: '', chunks: [] } },
  entities: { total: 0, types: [], top_entities: [], by_source: {}, examples: {} },
  feature_study: { documents: 0, positive: 0, negative: 0, duplicates_removed: 0, feature_sets: [], fractions: [], rows: [], best: null, reason: 'Needs more comments.' },
  similarity: { comments: 0, groups: 0, grouped_comments: 0, redundant: 0, compression: 0, redundant_pairs: 0, threshold: 0.3, mixed_groups: 0, rows: [], redundancy_note: '' },
  wsd: { words: 0, disambiguated: 0, ambiguous_words: [], model: { engine: 'fallback' }, rows: [], examples: [] },
};
api.renderPos(empty);
api.renderChunks(empty);
expect(nodes.get('chunk-empty').hidden === false, 'chunk empty note shown');
api.renderEntities(empty);
expect(nodes.get('entity-empty').hidden === false, 'entity empty note shown');
api.renderFeatureStudy(empty);
expect(nodes.get('study-empty').hidden === false, 'study empty note shown');
expect(nodes.get('study-empty').textContent === 'Needs more comments.', 'study reason surfaced');
api.renderSimilarity(empty);
expect(nodes.get('similarity-empty').hidden === false, 'similarity empty note shown');
api.renderWsd(empty);
expect(nodes.get('wsd-empty').hidden === false, 'wsd empty note shown');
expect(nodes.get('wsd-engine').textContent.includes('FALLBACK'), 'wsd fallback engine label');
expect(!nodes.get('wsd-note').innerHTML.includes('</p>'), 'no stray closing tag in wsd note');

console.log('\n== missing blocks entirely');
api.renderPos({});
api.renderChunks({});
api.renderEntities({});
api.renderFeatureStudy({});
api.renderSimilarity({});
api.renderWsd({});
console.log('  ok   no throw on missing keys');

console.log('\n== renderer ids exist in templates/index.html');
const html = fs.readFileSync('templates/index.html', 'utf8');
const ids = ['pos-tokens', 'pos-distribution', 'pos-compare', 'pos-keywords', 'pos-sentence',
  'chunk-count', 'chunk-types', 'chunk-empty', 'chunk-example',
  'entity-count', 'entity-types', 'entity-chips', 'entity-empty',
  'study-docs', 'study-chart', 'study-empty', 'study-insight-wrap', 'study-insight',
  'similarity-gain', 'similarity-stats', 'similarity-rows', 'similarity-empty',
  'wsd-engine', 'wsd-rows', 'wsd-empty', 'wsd-note',
  'deep-lab', 'deep-lab-note'];
for (const id of ids) expect(html.includes(`id="${id}"`), `id in HTML: ${id}`);

// Every selector the new renderers touch must exist in the markup.
const selectors = [...source.matchAll(/\$\('#([a-z-]+)'\)/g)].map((m) => m[1]);
const missing = [...new Set(selectors)].filter((id) => !html.includes(`id="${id}"`));
expect(missing.length === 0, `every $('#id') selector resolves in HTML${missing.length ? ` (missing: ${missing.join(', ')})` : ''}`);

console.log('\n== renderDashboard with fallback data');
try {
  api.renderDashboard(api.fallbackData);
  console.log('  ok   renderDashboard(fallbackData) did not throw');
  expect(nodes.get('pos-tokens').textContent.includes('TOKENS'), 'fallback pos rendered');
  expect(nodes.get('similarity-gain').textContent.includes('REDUCED'), 'fallback similarity rendered');
  expect(nodes.get('wsd-rows').innerHTML.includes('bank'), 'fallback wsd rendered');
  expect(nodes.get('study-chart').innerHTML.includes('Words only'), 'fallback study rendered');
} catch (e) { console.error('  FAIL renderDashboard:', e.message); process.exitCode = 1; }

if (process.exitCode) console.log('\nSOME CHECKS FAILED');
else console.log('\nALL JS RENDERER CHECKS PASSED');
