const state = {
  data: null,
  activeFilter: 'all',
  loadedComments: 5,
  particlesInitialized: false,
  observerActive: false,
  mouseX: 0,
  mouseY: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const fallbackData = {
  analyzed_at: 'just now',
  summary: 'Viewers are responding strongly to the practical walkthrough and the clarity of the examples. The clearest opportunity is to explain pricing and expose more advanced automation controls in a follow-up.',
  metrics: { comments: 4728, positive: 68, negative: 11, neutral: 21, engagement: 8.4, topics: 4 },
  sentiment: [
    { label: 'Positive', value: 68, count: 3216, color: 'lime' },
    { label: 'Neutral', value: 21, count: 994, color: 'slate' },
    { label: 'Negative', value: 11, count: 518, color: 'coral' },
  ],
  trend: [59, 64, 62, 71, 69, 75, 72, 79],
  trend_labels: ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'],
  topics: [
    { name: 'Product experience', share: 78, tone: 'positive', detail: 'Clear value and ease of use' },
    { name: 'Feature requests', share: 64, tone: 'neutral', detail: 'More control and personalization' },
    { name: 'Tutorial quality', share: 51, tone: 'positive', detail: 'Pacing praised by viewers' },
    { name: 'Pricing questions', share: 34, tone: 'negative', detail: 'Requests for simpler plans' },
  ],
  keywords: [
    { label: 'workflow', weight: 94 }, { label: 'simple', weight: 84 }, { label: 'feature', weight: 77 },
    { label: 'useful', weight: 71 }, { label: 'update', weight: 60 }, { label: 'creator', weight: 53 },
    { label: 'tutorial', weight: 48 }, { label: 'pricing', weight: 39 },
  ],
  morphology: {
    tokens_analyzed: 3184,
    distinct_forms: 612,
    distinct_roots: 498,
    inflected_tokens: 940,
    collapse_ratio: 18.6,
    forms: [
      { word: 'workflows', root: 'workflow', pos: 'Noun', form: 'Plural', affixes: ['s'], affix_details: [{ affix: '+s', kind: 'Inflectional', label: 'Plural' }], decomposition: 'workflow + s', count: 34, share: 1.1 },
      { word: 'tutorials', root: 'tutorial', pos: 'Noun', form: 'Plural', affixes: ['s'], affix_details: [{ affix: '+s', kind: 'Inflectional', label: 'Plural' }], decomposition: 'tutorial + s', count: 28, share: 0.9 },
      { word: 'viewers', root: 'view', pos: 'Noun', form: 'Plural', affixes: ['er', 's'], affix_details: [{ affix: '+er', kind: 'Derivational', label: 'Agent' }, { affix: '+s', kind: 'Inflectional', label: 'Plural' }], decomposition: 'view + er + s', count: 24, share: 0.8 },
      { word: 'recording', root: 'record', pos: 'Verb', form: 'Present participle', affixes: ['ing'], affix_details: [{ affix: '+ing', kind: 'Inflectional', label: 'Present participle' }], decomposition: 'record + ing', count: 19, share: 0.6 },
      { word: 'amazing', root: 'amaze', pos: 'Verb', form: 'Present participle', affixes: ['ing'], affix_details: [{ affix: '+ing', kind: 'Inflectional', label: 'Present participle' }], decomposition: 'amaze + ing', count: 16, share: 0.5 },
      { word: 'editing', root: 'edit', pos: 'Noun', form: 'Present participle', affixes: ['ing'], affix_details: [{ affix: '+ing', kind: 'Inflectional', label: 'Present participle' }], decomposition: 'edit + ing', count: 15, share: 0.5 },
      { word: 'updates', root: 'update', pos: 'Noun', form: 'Plural', affixes: ['s'], affix_details: [{ affix: '+s', kind: 'Inflectional', label: 'Plural' }], decomposition: 'update + s', count: 14, share: 0.4 },
      { word: 'features', root: 'feature', pos: 'Noun', form: 'Plural', affixes: ['s'], affix_details: [{ affix: '+s', kind: 'Inflectional', label: 'Plural' }], decomposition: 'feature + s', count: 12, share: 0.4 },
    ],
    affix_breakdown: [
      { affix: '+s', kind: 'Inflectional', label: 'Plural', count: 412, share: 12.9 },
      { affix: '+ing', kind: 'Inflectional', label: 'Present participle', count: 238, share: 7.5 },
      { affix: '+ed', kind: 'Inflectional', label: 'Past tense', count: 121, share: 3.8 },
      { affix: '+er', kind: 'Derivational', label: 'Agent', count: 96, share: 3.0 },
      { affix: '+or', kind: 'Derivational', label: 'Agent', count: 61, share: 1.9 },
      { affix: '+ly', kind: 'Derivational', label: 'Manner', count: 12, share: 0.4 },
    ],
  },
  ngrams: {
    documents: 4728,
    total_tokens: 3184,
    phrase_coverage: 64.2,
    unigrams: [
      { phrase: 'workflow', count: 341, share: 100 },
      { phrase: 'tutorial', count: 288, share: 84 },
      { phrase: 'simple', count: 201, share: 59 },
      { phrase: 'update', count: 164, share: 48 },
      { phrase: 'pricing', count: 121, share: 35 },
    ],
    bigrams: [
      { phrase: 'audio quality', count: 46, documents: 38, share: 100 },
      { phrase: 'clear walkthrough', count: 34, documents: 31, share: 74 },
      { phrase: 'practical examples', count: 27, documents: 24, share: 59 },
      { phrase: 'next update', count: 19, documents: 17, share: 41 },
      { phrase: 'channel average', count: 14, documents: 13, share: 30 },
      { phrase: 'pricing section', count: 11, documents: 10, share: 24 },
    ],
    trigrams: [
      { phrase: 'practical walkthrough examples', count: 12, documents: 11, share: 100 },
      { phrase: 'audio quality issues', count: 9, documents: 8, share: 75 },
      { phrase: 'next channel update', count: 6, documents: 6, share: 50 },
      { phrase: 'clear and simple', count: 4, documents: 4, share: 33 },
    ],
  },
  phrase_trends: [
    { phrase: 'audio quality', count: 46, documents: 38, series: [2, 3, 5, 6, 8, 9, 7, 6], momentum: 1.4, direction: 'rising', stance: 'negative' },
    { phrase: 'clear walkthrough', count: 34, documents: 31, series: [6, 5, 4, 4, 3, 4, 3, 2], momentum: -0.6, direction: 'cooling', stance: 'positive' },
    { phrase: 'next update', count: 19, documents: 17, series: [1, 1, 2, 3, 3, 4, 4, 5], momentum: 1.1, direction: 'rising', stance: 'positive' },
    { phrase: 'pricing section', count: 11, documents: 10, series: [3, 3, 2, 1, 1, 1, 0, 0], momentum: -0.8, direction: 'cooling', stance: 'negative' },
  ],
  pos: {
    tokens: 3184,
    sentences: 214,
    baseline: 'perceptron',
    distribution: [
      { group: 'Noun', count: 1180, share: 37.1 },
      { group: 'Adjective', count: 742, share: 23.3 },
      { group: 'Verb', count: 531, share: 16.7 },
      { group: 'Adverb', count: 302, share: 9.5 },
      { group: 'Determiner', count: 244, share: 7.7 },
      { group: 'Preposition', count: 121, share: 3.8 },
      { group: 'Conjunction', count: 64, share: 2.0 },
    ],
    tagger_comparison: [
      { tagger: 'perceptron', agreement: 100 },
      { tagger: 'rules', agreement: 54.2 },
      { tagger: 'hybrid', agreement: 100 },
    ],
    key_adjectives: ['clear', 'useful', 'quiet', 'broken', 'helpful', 'annoying'],
    key_nouns: ['audio', 'tutorial', 'microphone', 'recording', 'script', 'export'],
    key_verbs: ['explains', 'breaks', 'helps', 'runs', 'shows', 'sounds'],
    examples: {
      Noun: [
        { word: 'audio', tag: 'NN', group: 'Noun' },
        { word: 'quality', tag: 'NN', group: 'Noun' },
        { word: 'microphone', tag: 'NN', group: 'Noun' },
        { word: 'tutorial', tag: 'NN', group: 'Noun' },
      ],
    },
  },
  chunks: {
    total_chunks: 742,
    avg_per_sentence: 3.47,
    types: [
      { type: 'NP', label: 'Noun Phrase', count: 456, share: 61.5 },
      { type: 'VP', label: 'Verb Phrase', count: 212, share: 28.6 },
      { type: 'ADJP', label: 'Adjective Phrase', count: 74, share: 10.0 },
    ],
    phrases: [{ phrase: 'audio quality', length: 2, count: 46, comments: 38, share: 6.2, comment_share: 1.1 }],
    example: {
      text: 'the audio quality in this recording is really poor',
      chunks: [
        { label: 'Noun Phrase', text: 'the audio quality' },
        { label: 'Adjective Phrase', text: 'really poor' },
        { label: 'Verb Phrase', text: 'is' },
      ],
    },
  },
  entities: {
    total: 96,
    types: [
      { type: 'Technology', count: 34, share: 35.4 },
      { type: 'Product', count: 22, share: 22.9 },
      { type: 'Organization', count: 19, share: 19.8 },
      { type: 'Person', count: 13, share: 13.5 },
      { type: 'Location', count: 8, share: 8.3 },
    ],
    top_entities: [
      { text: 'Python', type: 'Technology', count: 18, share: 18.8 },
      { text: 'Microsoft', type: 'Organization', count: 12, share: 12.5 },
      { text: 'YouTube', type: 'Product', count: 9, share: 9.4 },
      { text: 'ChatGPT', type: 'Technology', count: 7, share: 7.3 },
      { text: 'Sarah Chen', type: 'Person', count: 5, share: 5.2 },
      { text: 'London', type: 'Location', count: 4, share: 4.2 },
    ],
    by_source: { gazetteer: 74, name_pattern: 14, context: 8 },
    examples: { Technology: [{ text: 'Python', sentence: 'I learned Python from this tutorial', source: 'gazetteer', confidence: 0.9 }] },
  },
  feature_study: {
    documents: 412,
    positive: 208,
    negative: 204,
    duplicates_removed: 26,
    feature_sets: [
      { key: 'unigrams', label: 'Words only' },
      { key: 'word_pos', label: 'Word + POS' },
      { key: 'chunks', label: 'Chunks only' },
      { key: 'all', label: 'Words + POS + chunks' },
    ],
    fractions: [0.25, 0.5, 0.75, 1.0],
    rows: [
      { feature_set: 'unigrams', label: 'Words only', fraction: 0.25, train_size: 103, accuracy: 71.2, f1: 70.4, vocabulary: 940 },
      { feature_set: 'unigrams', label: 'Words only', fraction: 0.5, train_size: 206, accuracy: 76.8, f1: 75.9, vocabulary: 1610 },
      { feature_set: 'unigrams', label: 'Words only', fraction: 0.75, train_size: 309, accuracy: 79.4, f1: 78.6, vocabulary: 2180 },
      { feature_set: 'unigrams', label: 'Words only', fraction: 1.0, train_size: 412, accuracy: 80.2, f1: 79.5, vocabulary: 2644 },
      { feature_set: 'word_pos', label: 'Word + POS', fraction: 0.25, train_size: 103, accuracy: 74.6, f1: 73.8, vocabulary: 1042 },
      { feature_set: 'word_pos', label: 'Word + POS', fraction: 0.5, train_size: 206, accuracy: 80.1, f1: 79.4, vocabulary: 1780 },
      { feature_set: 'word_pos', label: 'Word + POS', fraction: 0.75, train_size: 309, accuracy: 83.2, f1: 82.6, vocabulary: 2410 },
      { feature_set: 'word_pos', label: 'Word + POS', fraction: 1.0, train_size: 412, accuracy: 84.0, f1: 83.4, vocabulary: 2910 },
      { feature_set: 'chunks', label: 'Chunks only', fraction: 0.25, train_size: 103, accuracy: 66.1, f1: 65.2, vocabulary: 620 },
      { feature_set: 'chunks', label: 'Chunks only', fraction: 0.5, train_size: 206, accuracy: 70.4, f1: 69.6, vocabulary: 1080 },
      { feature_set: 'chunks', label: 'Chunks only', fraction: 0.75, train_size: 309, accuracy: 72.9, f1: 72.1, vocabulary: 1440 },
      { feature_set: 'chunks', label: 'Chunks only', fraction: 1.0, train_size: 412, accuracy: 74.1, f1: 73.4, vocabulary: 1710 },
      { feature_set: 'all', label: 'Words + POS + chunks', fraction: 0.25, train_size: 103, accuracy: 76.2, f1: 75.4, vocabulary: 1310 },
      { feature_set: 'all', label: 'Words + POS + chunks', fraction: 0.5, train_size: 206, accuracy: 82.0, f1: 81.3, vocabulary: 2210 },
      { feature_set: 'all', label: 'Words + POS + chunks', fraction: 0.75, train_size: 309, accuracy: 85.1, f1: 84.5, vocabulary: 2980 },
      { feature_set: 'all', label: 'Words + POS + chunks', fraction: 1.0, train_size: 412, accuracy: 86.3, f1: 85.7, vocabulary: 3510 },
    ],
    best: { feature_set: 'all', label: 'Words + POS + chunks', fraction: 1.0, train_size: 412, accuracy: 86.3, f1: 85.7, vocabulary: 3510 },
    reason: '',
  },
  similarity: {
    comments: 4728,
    groups: 812,
    grouped_comments: 2416,
    redundant: 1762,
    compression: 37.3,
    redundant_pairs: 3184,
    threshold: 0.3,
    mixed_groups: 96,
    rows: [
      { label: 'Audio Quality', size: 268, share: 6.4, sentiment: 'negative', sentiment_mix: { negative: 268 }, cohesion: 0.71, example: 'The audio quality is very poor' },
      { label: 'Tutorial Thank', size: 214, share: 5.1, sentiment: 'positive', sentiment_mix: { positive: 214 }, cohesion: 0.69, example: 'Great tutorial very helpful thanks' },
      { label: 'Microphone Sound', size: 187, share: 4.5, sentiment: 'mixed', sentiment_mix: { positive: 96, negative: 91 }, cohesion: 0.62, example: 'The microphone sounds great until the second half' },
      { label: 'Script Error', size: 143, share: 3.4, sentiment: 'negative', sentiment_mix: { negative: 143 }, cohesion: 0.66, example: 'The script does not run on my machine' },
    ],
    redundancy_note: '1762 of 4728 comments restate a point already made elsewhere.',
  },
  wsd: {
    words: 4,
    disambiguated: 26,
    ambiguous_words: ['bank', 'match', 'light', 'scale'],
    model: { trained: true, cached: true, words: 48, examples: 795, parameters: 80080, train_accuracy: 99.1, final_loss: 0.028, engine: 'GRU' },
    rows: [
      {
        word: 'bank',
        occurrences: 12,
        senses_used: 2,
        top_sense: 'depository_financial_institution.n.01',
        top_gloss: 'a financial institution that accepts deposits and channels the money it collects to borrowers',
        agreement: 0.58,
        breakdown: [
          { sense: 'depository_financial_institution.n.01', count: 7, share: 0.58 },
          { sense: 'bank.n.01', count: 5, share: 0.42 },
        ],
      },
      {
        word: 'match',
        occurrences: 9,
        senses_used: 2,
        top_sense: 'a person who stands out in a competition',
        top_gloss: 'a person who is skilful at a particular task',
        agreement: 0.44,
        breakdown: [
          { sense: 'contestant.n.01', count: 4, share: 0.44 },
          { sense: 'pairing.n.02', count: 5, share: 0.56 },
        ],
      },
    ],
    examples: [
      { word: 'bank', sense: 'depository_financial_institution.n.01', gloss: 'a financial institution that accepts deposits', confidence: 0.74 },
    ],
  },
  comments_feed: [
    { author: 'Maya Chen', text: 'This is the clearest walkthrough I have seen on this workflow. The pacing is perfect.', sentiment: 'positive', time: '2m ago', likes: '1.2k', initials: 'MC' },
    { author: 'Jordan Lee', text: 'Would love to see a deeper dive into the automation options in a future update.', sentiment: 'neutral', time: '8m ago', likes: '642', initials: 'JL' },
    { author: 'Ari Patel', text: 'The examples make the product feel much more approachable. Saving this for later.', sentiment: 'positive', time: '14m ago', likes: '438', initials: 'AP' },
    { author: 'Sam Rivera', text: 'The value is there, but the pricing section still feels a little hard to compare.', sentiment: 'negative', time: '22m ago', likes: '209', initials: 'SR' },
    { author: 'Noah Williams', text: 'The before and after section made the difference click for me instantly.', sentiment: 'positive', time: '31m ago', likes: '177', initials: 'NW' },
  ],
};

function formatNumber(value) {
  return new Intl.NumberFormat('en-US').format(Math.round(value));
}

function animateNumber(element, target, duration = 1200, suffix = '') {
  if (!element) return;
  const start = performance.now();
  const initial = Number(element.dataset.current || 0);
  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 4);
    const value = initial + (target - initial) * eased;
    element.textContent = `${formatNumber(value)}${suffix}`;
    if (progress < 1) requestAnimationFrame(frame);
    else element.dataset.current = target;
  }
  requestAnimationFrame(frame);
}

function renderMetrics(data) {
  animateNumber($('.count-value'), data.metrics.comments);
  animateNumber($('#positive-value'), data.metrics.positive);
  $('#positive-count').textContent = `${formatNumber(data.metrics.comments * data.metrics.positive / 100)} comments`;
  $('#topic-value').textContent = data.metrics.topics;
  $('.metric-card:nth-child(3) .metric-value').innerHTML = `${data.metrics.engagement.toFixed(1)}<small>x</small>`;
  setTimeout(() => {
    $('.metric-card:nth-child(2) .mini-bar i').style.width = `${data.metrics.positive}%`;
  }, 200);
  $('#donut-value').textContent = `${data.metrics.positive}%`;
  $('#analyzed-time').textContent = `Last analyzed · ${data.analyzed_at || 'just now'}`;
}

function renderSentiment(data) {
  const total = data.sentiment.reduce((sum, item) => sum + item.value, 0);
  let cursor = 0;
  const stops = data.sentiment.map((item) => {
    const start = cursor;
    cursor += (item.value / total) * 100;
    const color = item.color === 'lime' ? 'var(--accent)' : item.color === 'coral' ? 'var(--coral)' : '#636c69';
    return `${color} ${start}% ${cursor}%`;
  }).join(', ');
  $('#donut-chart').style.background = `conic-gradient(${stops})`;
  $('#sentiment-legend').innerHTML = data.sentiment.map((item, i) => {
    const colorClass = item.color === 'lime' ? 'lime-dot' : item.color === 'coral' ? 'coral-dot' : 'slate-dot';
    return `<div class="legend-item" style="animation-delay: ${i * 0.1}s"><span class="signal-dot ${colorClass}"></span><span class="legend-name">${item.label}</span><strong>${item.value}%</strong></div>`;
  }).join('');
}

function chartPointPath(values) {
  const width = 720;
  const height = 204;
  const paddingX = 12;
  const max = Math.max(...values) + 6;
  const min = Math.min(...values) - 6;
  const usableWidth = width - paddingX * 2;
  const usableHeight = height - 12;
  return values.map((value, index) => {
    const x = paddingX + (index / Math.max(values.length - 1, 1)) * usableWidth;
    const y = 9 + (1 - (value - min) / Math.max(max - min, 1)) * usableHeight;
    return [x, y];
  });
}

function renderChart(data) {
  const points = chartPointPath(data.trend);
  const line = points.map(([x, y], index) => `${index ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
  const area = `${line} L ${points[points.length - 1][0].toFixed(2)} 218 L ${points[0][0].toFixed(2)} 218 Z`;
  const lineEl = $('#chart-line');
  lineEl.setAttribute('d', line);
  lineEl.style.animation = 'none';
  lineEl.offsetHeight;
  lineEl.style.animation = 'drawLine 2s var(--ease-out) forwards';
  $('#chart-area').setAttribute('d', area);
  $('#chart-points').innerHTML = points.map(([x, y], index) =>
    `<circle class="chart-point" cx="${x}" cy="${y}" r="${index === points.length - 1 ? 4 : 2.8}" style="animation-delay: ${index * 0.1 + 0.5}s"></circle>`
  ).join('');
  $('#chart-labels').innerHTML = data.trend_labels.map((label) => `<span>${label}</span>`).join('');
  const lift = ((data.trend[data.trend.length - 1] - data.trend[0]) / data.trend[0]) * 100;
  $('#trend-score').textContent = `${lift >= 0 ? '+' : ''}${lift.toFixed(1)}%`;
}

function renderTopics(data) {
  $('#topic-list').innerHTML = data.topics.map((topic, i) => {
    const toneClass = topic.tone === 'positive' ? 'lime-dot' : topic.tone === 'negative' ? 'coral-dot' : 'slate-dot';
    return `<div class="topic-row" style="animation-delay: ${i * 0.1}s"><div class="topic-main"><div class="topic-name"><span class="signal-dot ${toneClass}"></span>${topic.name}</div><span class="topic-detail">${topic.detail}</span><span class="topic-bar"><i style="width:${topic.share}%"></i></span></div><span class="topic-share">${topic.share}%</span></div>`;
  }).join('');
}

function renderKeywords(data) {
  $('#keyword-cloud').innerHTML = data.keywords.map((word, i) =>
    `<span class="keyword" style="opacity:${0.58 + word.weight / 260}; animation-delay: ${i * 0.06}s">${word.label}</span>`
  ).join('');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

function renderMorphology(data) {
  const morphology = data.morphology;
  if (!morphology) return;

  $('#morphology-count').textContent = `${morphology.distinct_forms} FORMS`;
  $('#lab-note').textContent = `${morphology.tokens_analyzed} tokens · ${morphology.collapse_ratio}% collapsed to roots`;

  $('#affix-stats').innerHTML = (morphology.affix_breakdown || []).map((entry) =>
    `<span class="affix-stat"><b>${escapeHtml(entry.affix)}</b><span>${escapeHtml(entry.label)}</span><span class="affix-kind">${escapeHtml(entry.kind)}</span></span>`
  ).join('');

  const rows = morphology.forms || [];
  $('#morphology-rows').innerHTML = rows.map((entry) => {
    const affixes = (entry.affix_details || []).map((detail) =>
      detail.kind === 'Derivational'
        ? `<span class="derivational">${escapeHtml(detail.affix)}</span>`
        : escapeHtml(detail.affix)
    ).join(' ') || '—';
    const width = Math.max(2, Math.min(100, (entry.share || 0) * 8));
    return `<tr>
      <td class="form-word">${escapeHtml(entry.word)}</td>
      <td class="root-word">${escapeHtml(entry.root)}</td>
      <td class="affix-cell">${affixes}</td>
      <td class="affix-cell">${escapeHtml(entry.pos)}</td>
      <td class="num hits">${entry.count}<span class="share-bar"><i style="width:${width}%"></i></span></td>
    </tr>`;
  }).join('');

  $('#morphology-empty').hidden = rows.length > 0;
  $('#morphology-insight').innerHTML = `<strong>${morphology.distinct_roots} roots from ${morphology.distinct_forms} forms.</strong> ${morphology.inflected_tokens} inflected tokens were normalized to their root before keyword extraction.`;
}

function renderNgrams(data) {
  const ngrams = data.ngrams;
  if (!ngrams) return;

  $('#ngram-coverage').textContent = `${ngrams.phrase_coverage}% COVERAGE`;

  const fill = (listId, rows) => {
    $(listId).innerHTML = (rows || []).map((row, index) => {
      const width = Math.max(3, Math.min(100, row.share || 0));
      return `<div class="ngram-row" style="animation-delay: ${index * 0.06}s">
        <div class="ngram-row-head">
          <span class="ngram-phrase">${escapeHtml(row.phrase)}</span>
          <span class="ngram-count">${row.count}×</span>
        </div>
        <span class="ngram-bar"><i style="width:${width}%"></i></span>
      </div>`;
    }).join('');
  };

  fill('#bigram-list', ngrams.bigrams);
  fill('#trigram-list', ngrams.trigrams);
  $('#ngram-empty').hidden = Boolean((ngrams.bigrams || []).length || (ngrams.trigrams || []).length);
}

function renderPhraseTrends(data) {
  const trends = data.phrase_trends || [];
  $('#trend-phrase-note').textContent = trends.length ? `${trends.length} TRACKED` : '—';
  $('#trend-empty').hidden = trends.length > 0;

  $('#phrase-trend-list').innerHTML = trends.map((trend, index) => {
    const series = trend.series || [];
    const peak = Math.max(...series, 1);
    const bars = series.map((value) => {
      const height = Math.max(3, Math.round((value / peak) * 100));
      return `<i class="${value > 0 ? 'has-hit' : ''}" style="height:${height}%" title="${value} mentions"></i>`;
    }).join('');
    const arrow = trend.direction === 'rising' ? '↗' : trend.direction === 'cooling' ? '↘' : '→';
    return `<div class="phrase-trend" style="animation-delay: ${index * 0.08}s">
      <div class="phrase-trend-head">
        <span class="phrase-trend-phrase">${escapeHtml(trend.phrase)}</span>
        <span class="phrase-trend-meta">
          <span class="trend-stance ${escapeHtml(trend.stance)}">${escapeHtml(trend.stance)}</span>
          <span class="trend-dir ${escapeHtml(trend.direction)}">${arrow} ${escapeHtml(trend.direction)}</span>
        </span>
      </div>
      <div class="spark-bars">${bars}</div>
    </div>`;
  }).join('');
}

async function runWordForms(event) {
  event?.preventDefault();
  const input = $('#generator-input');
  const message = $('#generator-message');
  const button = $('#generator-form button');
  const word = input.value.trim().toLowerCase();

  if (!word) {
    message.textContent = 'Enter a word to generate its forms.';
    return;
  }

  button.disabled = true;
  message.textContent = '';
  try {
    const response = await fetch(`/api/morphology/forms?word=${encodeURIComponent(word)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not analyze that word.');

    const analysis = data.analysis;
    const forms = data.forms || [];
    $('#generator-word').textContent = analysis.word;
    $('#generator-decomposition').textContent = `${analysis.decomposition} · ${analysis.pos}`;
    $('#generator-forms').innerHTML = forms.map((form) =>
      `<span class="form-chip ${form.label === 'Base form' ? 'is-base' : ''}">${escapeHtml(form.form)}${form.label === 'Base form' ? '' : `<small>${escapeHtml(form.label)}</small>`}</span>`
    ).join('');
    $('#generator-note').textContent = analysis.root && analysis.root !== analysis.word
      ? `Built from the root "${analysis.root}" (${analysis.form.toLowerCase()}).`
      : `Base form of a ${analysis.pos.toLowerCase()}.`;
    $('#generator-result').hidden = false;
  } catch (error) {
    message.textContent = error.message;
    $('#generator-result').hidden = true;
  } finally {
    button.disabled = false;
  }
}

function renderComments(data) {
  const filtered = state.activeFilter === 'all' ? data.comments_feed : data.comments_feed.filter((comment) => comment.sentiment === state.activeFilter);
  $('#comments-list').innerHTML = filtered.slice(0, state.loadedComments).map((comment, index) =>
    `<div class="comment-item" style="animation-delay: ${index * 80}ms"><span class="comment-avatar">${comment.initials}</span><div class="comment-body"><div class="comment-meta"><span class="comment-author">${comment.author}</span><span class="comment-time">${comment.time}</span><span class="comment-likes">· ${comment.likes} likes</span></div><p class="comment-text">${comment.text}</p></div><span class="sentiment-badge ${comment.sentiment}">${comment.sentiment}</span></div>`
  ).join('');
  $('#load-more').style.display = filtered.length > state.loadedComments ? 'block' : 'none';
}

function renderPos(data) {
  const pos = data.pos;
  if (!pos) return;

  $('#pos-tokens').textContent = `${formatNumber(pos.tokens)} TOKENS`;

  const distribution = pos.distribution || [];
  $('#pos-distribution').innerHTML = distribution.slice(0, 7).map((entry) => `
    <div class="pos-row">
      <div class="pos-row-head">
        <span class="pos-name">${escapeHtml(entry.group)}</span>
        <span class="pos-meta">${entry.share}%</span>
      </div>
      <div class="pos-bar"><i style="width:${Math.max(2, entry.share)}%"></i></div>
    </div>`).join('');

  const compared = (pos.tagger_comparison || []).map((entry) => {
    const percent = Math.round(entry.agreement || 0);
    const isBase = entry.tagger === pos.baseline;
    return `
    <div class="pos-row">
      <div class="pos-row-head">
        <span class="pos-name">${escapeHtml(entry.tagger)}${isBase ? ' (baseline)' : ''}</span>
        <span class="pos-meta">${percent}% agree</span>
      </div>
      <div class="pos-bar"><i style="width:${Math.max(2, percent)}%"></i></div>
    </div>`;
  }).join('');
  $('#pos-compare').innerHTML = compared;

  const chips = [
    ...(pos.key_adjectives || []).slice(0, 3).map((w) => [w, 'ADJ']),
    ...(pos.key_nouns || []).slice(0, 3).map((w) => [w, 'NOUN']),
    ...(pos.key_verbs || []).slice(0, 3).map((w) => [w, 'VERB']),
  ].map(([word, tag]) =>
    `<span class="form-chip"><b>${escapeHtml(word)}</b><small>${tag}</small></span>`).join('');
  $('#pos-keywords').innerHTML = chips;

  // A worked example of tagged words, the same shape the documentation uses.
  const sample = pos.examples || {};
  const sampleGroup = ['Noun', 'Adjective', 'Verb'].find((group) => (sample[group] || []).length);
  if (sampleGroup) {
    $('#pos-sentence').innerHTML = sample[sampleGroup].slice(0, 10).map((entry) =>
      `<span class="tag-token">${escapeHtml(entry.word)}<em>${escapeHtml(entry.tag)}</em></span>`).join('');
  }
}

function renderChunks(data) {
  const chunks = data.chunks;
  if (!chunks) return;

  $('#chunk-count').textContent = `${formatNumber(chunks.total_chunks)} PHRASES`;

  const types = chunks.types || [];
  $('#chunk-types').innerHTML = types.map((entry) => `
    <div class="chunk-type">
      <div class="chunk-type-head">
        <span class="chunk-type-name">${escapeHtml(entry.label)}</span>
        <span class="chunk-type-meta">${entry.share}%</span>
      </div>
      <div class="chunk-bar"><i style="width:${Math.max(2, entry.share)}%"></i></div>
    </div>`).join('');
  $('#chunk-empty').hidden = types.length > 0;

  const example = chunks.example;
  if (example && example.chunks && example.chunks.length) {
    $('#chunk-example').innerHTML = example.chunks.slice(0, 6).map((phrase) =>
      `<span class="chunk-phrase">${escapeHtml(phrase.text)}<b>${escapeHtml(phrase.label)}</b></span>`).join('');
  }
}

function renderEntities(data) {
  const entities = data.entities;
  if (!entities) return;

  $('#entity-count').textContent = `${entities.total} MENTIONS`;

  const types = entities.types || [];
  $('#entity-types').innerHTML = types.map((entry) => `
    <div class="chunk-type">
      <div class="chunk-type-head">
        <span class="chunk-type-name">${escapeHtml(entry.type)}</span>
        <span class="chunk-type-meta">${entry.count} · ${entry.share}%</span>
      </div>
      <div class="chunk-bar"><i style="width:${Math.max(2, entry.share)}%"></i></div>
    </div>`).join('');

  const chips = (entities.top_entities || []).slice(0, 10).map((entry) => `
    <span class="entity-chip entity-type ${escapeHtml(entry.type.toLowerCase())}">
      <b>${escapeHtml(entry.text)}</b><span>${entry.count}× ${escapeHtml(entry.type)}</span>
    </span>`).join('');
  $('#entity-chips').innerHTML = chips;
  $('#entity-empty').hidden = (entities.top_entities || []).length > 0;
}

const FEATURE_COLORS = {
  unigrams: 'var(--blue)',
  word_pos: 'var(--violet)',
  chunks: 'var(--coral)',
  all: 'var(--accent)',
};

function renderFeatureStudy(data) {
  const study = data.feature_study;
  if (!study) return;

  const rows = study.rows || [];
  $('#study-docs').textContent = `${study.documents || 0} COMMENTS`;

  if (!rows.length) {
    $('#study-empty').hidden = false;
    $('#study-empty').textContent = study.reason || 'Not enough data to compare feature sets.';
    $('#study-insight').innerHTML = `<strong>Awaiting a larger sample.</strong> ${escapeHtml(study.reason || '')}`;
    return;
  }

  const best = study.best || {};
  const sets = study.feature_sets || [];
  const sizes = [...new Set(rows.map((row) => row.train_size))].sort((a, b) => a - b);
  const peak = Math.max(...rows.map((row) => row.f1), 1);

  $('#study-chart').innerHTML = sets.map((set) => {
    const mine = rows.filter((row) => row.feature_set === set.key);
    return `
    <div class="study-series">
      <div class="study-series-head">
        <span class="study-series-name">${escapeHtml(set.label)}</span>
        <span class="study-legend">${sizes.length} sizes</span>
      </div>
      <div class="study-points">
        ${sizes.map((size) => {
          const row = mine.find((entry) => entry.train_size === size) || {};
          const f1 = row.f1 || 0;
          const isBest = best.feature_set === set.key && best.train_size === size;
          return `<div class="study-point" title="${escapeHtml(set.label)} @ ${size}: F1 ${f1.toFixed(1)}">
            <div class="study-point-bar${isBest ? ' best' : ''}" style="height:${Math.max(3, (f1 / peak) * 100)}%;${isBest ? '' : `background:${FEATURE_COLORS[set.key] || 'var(--blue)'}`}"></div>
            <span class="study-point-label">${size}</span>
          </div>`;
        }).join('')}
      </div>
    </div>`;
  }).join('');

  const hasBest = best.feature_set !== undefined;
  const detail = hasBest
    ? `<strong>Best combination:</strong> ${escapeHtml(best.label)} reached F1 ${Number(best.f1).toFixed(1)} on ${best.train_size} comments.`
    : '<strong>No clear winner yet.</strong>';
  $('#study-insight').innerHTML = `${detail} ${escapeHtml(study.duplicates_removed || 0)} duplicate comments were collapsed before scoring.`;
}

function renderSimilarity(data) {
  const similarity = data.similarity;
  if (!similarity) return;

  $('#similarity-gain').textContent = `${similarity.compression}% REDUCED`;

  const stats = [
    ['DISCUSSIONS', similarity.groups],
    ['COMMENTS GROUPED', similarity.grouped_comments],
    ['REPEATED PAIRS', similarity.redundant_pairs],
  ].map(([label, value]) =>
    `<span class="affix-stat"><b>${value}</b><span>${label}</span></span>`).join('');
  $('#similarity-stats').innerHTML = stats;

  const rows = similarity.rows || [];
  const peak = Math.max(...rows.map((row) => row.size), 1);
  $('#similarity-rows').innerHTML = rows.map((row) => `
    <div class="discussion">
      <div class="discussion-head">
        <span class="discussion-topic">${escapeHtml(row.label)}</span>
        <span class="discussion-meta">
          <span class="discussion-stance ${escapeHtml(row.sentiment)}">${escapeHtml(row.sentiment)}</span>
          ${row.size} comments
        </span>
      </div>
      <div class="discussion-quote">${escapeHtml(row.example)}</div>
      <div class="discussion-bar"><i style="width:${Math.max(3, (row.size / peak) * 100)}%"></i></div>
    </div>`).join('');
  $('#similarity-empty').hidden = rows.length > 0;
}

function renderWsd(data) {
  const wsd = data.wsd;
  if (!wsd) return;

  const model = wsd.model || {};
  $('#wsd-engine').textContent = model.engine === 'GRU' ? 'GRU · READY' : 'FALLBACK';

  const rows = wsd.rows || [];
  $('#wsd-rows').innerHTML = rows.map((row) => {
    const top = (row.breakdown || [])[0];
    const share = Math.round((top ? top.share : 0) * 100);
    return `
    <div class="wsd-item">
      <div class="wsd-item-head">
        <span class="wsd-word">${escapeHtml(row.word)}</span>
        <span class="wsd-confidence">${row.occurrences}× · ${share}% ${escapeHtml(row.top_sense)}</span>
      </div>
      <span class="wsd-gloss">${escapeHtml(row.top_gloss)}</span>
      <div class="wsd-spread">
        ${(row.breakdown || []).map((sense, index) =>
          `<i class="${index === 0 ? 'top' : ''}" style="flex:${Math.max(0.05, sense.share)}"></i>`).join('')}
      </div>
    </div>`;
  }).join('');
  $('#wsd-empty').hidden = rows.length > 0;

  if (model.engine !== 'GRU') {
    $('#wsd-note').innerHTML = '<strong>Model still training.</strong> The app is scoring senses with the gloss baseline until the GRU finishes its first run.';
  } else if (rows.length) {
    const widest = wsd.rows.reduce((a, b) => (a.senses_used > b.senses_used ? a : b));
    $('#wsd-note').innerHTML = `<strong>One word, ${widest.senses_used} meanings.</strong> "${escapeHtml(widest.word)}" was read as ${widest.senses_used} different WordNet senses in this section, so counting the bare string would have merged them.`;
  }
}

function renderDashboard(data) {
  state.data = data;
  renderMetrics(data);
  renderSentiment(data);
  renderChart(data);
  renderTopics(data);
  renderKeywords(data);
  renderMorphology(data);
  renderNgrams(data);
  renderPhraseTrends(data);
  renderPos(data);
  renderChunks(data);
  renderEntities(data);
  renderFeatureStudy(data);
  renderSimilarity(data);
  renderWsd(data);
  $('#summary-text').textContent = data.summary;
  state.loadedComments = 5;
  renderComments(data);
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove('show'), 2800);
}

async function runAnalysis(event) {
  event.preventDefault();
  const form = $('#analyze-form');
  const card = $('.analyzer-card');
  const button = form.querySelector('button');
  const label = $('#button-label');
  const message = $('#form-message');
  card.classList.add('is-loading');
  button.disabled = true;
  label.textContent = 'Reading the room...';
  message.textContent = '';
  try {
    const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: $('#youtube-url').value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to analyze this URL.');
    await new Promise((resolve) => setTimeout(resolve, 620));
    renderDashboard(data);
    showToast('Fresh signal loaded into the workspace.');
  } catch (error) {
    message.textContent = error.message;
  } finally {
    card.classList.remove('is-loading');
    button.disabled = false;
    label.textContent = 'Analyze comments';
  }
}

function setupTheme() {
  const root = document.documentElement;
  const saved = localStorage.getItem('signal-theme');
  if (saved) root.dataset.theme = saved;
  $('#theme-toggle').addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    localStorage.setItem('signal-theme', next);
    $('#theme-toggle').setAttribute('aria-label', `Switch to ${next === 'dark' ? 'light' : 'dark'} theme`);
  });
}

function setupInteractions() {
  $('#analyze-form').addEventListener('submit', runAnalysis);
  $('#generator-form').addEventListener('submit', runWordForms);
  $$('.filter-button').forEach((button) => button.addEventListener('click', () => {
    $$('.filter-button').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    state.activeFilter = button.dataset.filter;
    state.loadedComments = 5;
    renderComments(state.data || fallbackData);
  }));
  $('#load-more').addEventListener('click', () => {
    state.loadedComments += 5;
    renderComments(state.data || fallbackData);
  });
  $('#copy-summary').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($('#summary-text').textContent);
      showToast('Summary copied to clipboard.');
    } catch {
      showToast('Summary ready to copy.');
    }
  });
  $('#export-button').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(state.data || fallbackData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'signal-lab-snapshot.json';
    link.click();
    URL.revokeObjectURL(url);
    showToast('Snapshot exported as JSON.');
  });
}

function setupCursorGlow() {
  const glow = document.createElement('div');
  glow.className = 'cursor-glow';
  document.body.appendChild(glow);
  let ticking = false;
  document.addEventListener('mousemove', (e) => {
    state.mouseX = e.clientX;
    state.mouseY = e.clientY;
    if (!ticking) {
      requestAnimationFrame(() => {
        glow.style.left = e.clientX + 'px';
        glow.style.top = e.clientY + 'px';
        glow.classList.add('active');
        ticking = false;
      });
      ticking = true;
    }
  });
  document.addEventListener('mouseleave', () => glow.classList.remove('active'));
}

function setupParticles() {
  if (state.particlesInitialized) return;
  state.particlesInitialized = true;
  const canvas = document.createElement('canvas');
  canvas.id = 'particle-canvas';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  let particles = [];
  const PARTICLE_COUNT = 50;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      z: Math.random() * 200,
      size: Math.random() * 2 + 0.5,
      speedX: (Math.random() - 0.5) * 0.4,
      speedY: (Math.random() - 0.5) * 0.3,
      speedZ: (Math.random() - 0.5) * 0.5,
      opacity: Math.random() * 0.3 + 0.05,
      hue: Math.random() > 0.7 ? 85 : 210,
    });
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach((p) => {
      p.x += p.speedX;
      p.y += p.speedY;
      p.z += p.speedZ;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
      if (p.z < -100) p.z = 200;
      if (p.z > 200) p.z = -100;
      const scale = (p.z + 100) / 300;
      const finalSize = p.size * (0.5 + scale);
      const finalOpacity = p.opacity * (0.3 + scale * 0.7);
      ctx.beginPath();
      ctx.arc(p.x, p.y, finalSize, 0, Math.PI * 2);
      ctx.fillStyle = p.hue === 85
        ? `rgba(156, 255, 0, ${finalOpacity})`
        : `rgba(137, 183, 255, ${finalOpacity})`;
      ctx.fill();
    });
    requestAnimationFrame(animate);
  }
  animate();
}

function setupScrollReveal() {
  if (state.observerActive) return;
  state.observerActive = true;
  const sections = $$('.reveal');
  if (!('IntersectionObserver' in window)) {
    sections.forEach((s) => s.classList.add('visible'));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.remove('hidden');
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
  sections.forEach((section) => {
    section.classList.add('hidden');
    observer.observe(section);
  });
}

function setup3DParallax() {
  const orbit = $('.hero-orbit');
  const heroSection = $('.hero-section');
  if (!orbit || !heroSection) return;

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    const maxScroll = 600;
    if (scrollY < maxScroll) {
      const progress = scrollY / maxScroll;
      const orbitZ = 40 - progress * 80;
      const orbitRotateX = progress * 15;
      orbit.style.transform = `translateZ(${orbitZ}px) rotateX(${orbitRotateX}deg) scale(${1 - progress * 0.1})`;
      orbit.style.opacity = 1 - progress * 0.6;
    }
  }, { passive: true });

  heroSection.addEventListener('mousemove', (e) => {
    const rect = heroSection.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    const heroCopy = heroSection.querySelector('.hero-copy');
    if (heroCopy) {
      heroCopy.style.transform = `translateZ(40px) rotateY(${x * 3}deg) rotateX(${-y * 3}deg)`;
    }
  });
  heroSection.addEventListener('mouseleave', () => {
    const heroCopy = heroSection.querySelector('.hero-copy');
    if (heroCopy) {
      heroCopy.style.transform = 'translateZ(40px)';
    }
  });
}

function setup3DCardTilt() {
  const cards = $$('.metric-card');
  cards.forEach((card) => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      const rotateY = x * 15;
      const rotateX = -y * 15;
      const translateZ = 20;
      card.style.transform = `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(${translateZ}px)`;
      card.style.setProperty('--mouse-x', `${(e.clientX - rect.left) / rect.width * 100}%`);
      card.style.setProperty('--mouse-y', `${(e.clientY - rect.top) / rect.height * 100}%`);
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
}

function setup3DPanelTilt() {
  const panels = $$('.panel');
  panels.forEach((panel) => {
    panel.addEventListener('mousemove', (e) => {
      const rect = panel.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      const rotateY = x * 6;
      const rotateX = -y * 6;
      panel.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(8px)`;
    });
    panel.addEventListener('mouseleave', () => {
      panel.style.transform = '';
    });
  });
}

function setup3DAnalyzerCard() {
  const card = $('.analyzer-card');
  if (!card) return;
  card.addEventListener('mousemove', (e) => {
    const rect = card.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    card.style.setProperty('--mouse-x', `${x * 100}%`);
    card.style.setProperty('--mouse-y', `${y * 100}%`);
    const rotateY = (x - 0.5) * 4;
    const rotateX = -(y - 0.5) * 4;
    card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(20px)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = 'translateZ(20px)';
  });
}

function setupTypingEffect() {
  const h1 = $('#page-title');
  if (!h1) return;
  const emContent = h1.querySelector('em');
  if (!emContent) return;
  const emText = emContent.textContent;
  const fullText = h1.textContent;
  h1.innerHTML = '';
  const textBefore = fullText.split(emText)[0];
  const textSpan = document.createElement('span');
  textSpan.textContent = textBefore;
  h1.appendChild(textSpan);
  const emSpan = document.createElement('em');
  emSpan.textContent = '';
  h1.appendChild(emSpan);
  const cursor = document.createElement('span');
  cursor.style.cssText = 'display:inline-block;width:2px;height:0.9em;background:var(--accent);margin-left:2px;animation:blink 1s step-end infinite;vertical-align:text-bottom;';
  h1.appendChild(cursor);
  const style = document.createElement('style');
  style.textContent = '@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}';
  document.head.appendChild(style);
  let charIndex = 0;
  function typeChar() {
    if (charIndex < emText.length) {
      emSpan.textContent += emText[charIndex];
      charIndex++;
      setTimeout(typeChar, 60 + Math.random() * 40);
    } else {
      setTimeout(() => cursor.remove(), 1200);
    }
  }
  setTimeout(typeChar, 800);
}

document.addEventListener('DOMContentLoaded', () => {
  setupTheme();
  setupInteractions();
  setupCursorGlow();
  setupParticles();
  setupScrollReveal();
  setup3DParallax();
  setup3DCardTilt();
  setup3DPanelTilt();
  setup3DAnalyzerCard();
  renderDashboard(fallbackData);
  runWordForms();
  setupTypingEffect();
});
