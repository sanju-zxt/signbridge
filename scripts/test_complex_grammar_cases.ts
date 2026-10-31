import { parseEnglishToISLSentenceIR } from '../packages/types/src/islGrammar';

const testSentences = [
  { id: 1, category: 'Past, present, future time', text: "Yesterday I went to the hospital, tomorrow I will go to school." },
  { id: 2, category: 'Negation', text: "I do not want medicine, I want water." },
  { id: 3, category: 'Questions', text: "Why are you calling the police when there is no danger?" },
  { id: 4, category: 'Multiple clauses', text: "When you come home, please call me." },
  { id: 5, category: 'Conditional sentences', text: "If you need help, where should you go?" },
  { id: 6, category: 'Pronouns and spatial references', text: "You stay here on the left, I will go to the right." },
  { id: 7, category: 'Relationships between people', text: "My friend is deaf, but the doctor is hearing." },
  { id: 8, category: 'Complex requests', text: "Please wait here and do not leave until the doctor comes." },
  { id: 9, category: 'Emergency communication', text: "Severe danger here, please call an ambulance immediately!" },
  { id: 10, category: 'Context-dependent expressions', text: "Nice to meet you, do you understand Indian Sign Language?" },
];

console.log('Testing 10 Complex Linguistic Sentences:\n');
for (const s of testSentences) {
  const ir = parseEnglishToISLSentenceIR(s.text);
  console.log(`[${s.id}] Category: ${s.category}`);
  console.log(`    English: "${s.text}"`);
  console.log(`    Intent:  ${ir.intent}`);
  console.log(`    Clauses: ${ir.clauses.length}`);
  console.log(`    Glosses: ${ir.glossSequence.join(' ')}`);
  console.log(`    NMMs:    ${ir.nonManualSummary.join(', ')}`);
  console.log(`    Roles:   ${ir.linearSigns.map(sig => `${sig.role}:${sig.gloss}`).join(' -> ')}`);
  console.log('-'.repeat(80));
}
