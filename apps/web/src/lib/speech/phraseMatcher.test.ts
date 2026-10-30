import { matchSpeechPhrases, getPrimaryActionFromSpeech, matchSpeechSequence } from './phraseMatcher';

function assertEqual(actual: any, expected: any, testName: string) {
  const actualStr = JSON.stringify(actual);
  const expectedStr = JSON.stringify(expected);
  if (actualStr === expectedStr) {
    console.log(`✅ PASS: ${testName}`);
  } else {
    console.error(`❌ FAIL: ${testName}\n  Expected: ${expectedStr}\n  Got:      ${actualStr}`);
    process.exit(1);
  }
}

console.log('--- Testing Phrase Matcher (25 ISL Concepts) ---');

// Test 1: "thank you so much" should map to THANK_YOU with high priority
const m1 = getPrimaryActionFromSpeech('Thank you so much for your assistance');
assertEqual(m1, 'THANK_YOU', 'Multi-word Thank You mapping');

// Test 2: "sorry" should map to SORRY
const m2 = getPrimaryActionFromSpeech("I'm so sorry about that");
assertEqual(m2, 'SORRY', 'Apology phrase mapping');

// Test 3: "crying" should map to CRYING
const m3 = getPrimaryActionFromSpeech('The child is crying loudly');
assertEqual(m3, 'CRYING', 'Emotional crying phrase mapping');

// Test 4: "going" should map to GOING
const m4 = getPrimaryActionFromSpeech("I'm going now, see you later");
assertEqual(m4, 'GOING', 'Going / departing phrase mapping');

// Test 5: "good morning" should map to HI / greeting
const m5 = getPrimaryActionFromSpeech('Good morning everyone!');
assertEqual(m5, 'HI', 'Greeting mapping');

// Test 6: "namaste" should map to HELLO
const m6 = getPrimaryActionFromSpeech('Namaste, how are you?');
assertEqual(m6, 'HELLO', 'Indic greeting mapping');

// Test 7: Word boundary protection — word like "crying" vs "descrying" (no false positives)
const m7 = getPrimaryActionFromSpeech('The descrying telescope');
assertEqual(m7, null, 'Word boundary prevents substring match in descrying');

// Test 8: Multiple phrase prioritization in a single sentence
const m8 = matchSpeechPhrases('Hello and thank you so much');
assertEqual(m8[0].canonicalAction, 'THANK_YOU', 'Priority: multi-word thank you evaluated before single hello');
assertEqual(m8[1].canonicalAction, 'HELLO', 'Second action recognized in queue');

// Test 9: 25-Sign concepts: HOSPITAL, BATHROOM, POLICE, EMERGENCY, GOOD, STOP
assertEqual(getPrimaryActionFromSpeech('Where is the nearest hospital?'), 'HOSPITAL', 'Hospital mapping');
assertEqual(getPrimaryActionFromSpeech('Where is the washroom?'), 'BATHROOM', 'Bathroom mapping');
assertEqual(getPrimaryActionFromSpeech('Call the police right now!'), 'POLICE', 'Police mapping');
assertEqual(getPrimaryActionFromSpeech('We have an emergency situation!'), 'EMERGENCY', 'Emergency mapping');
assertEqual(getPrimaryActionFromSpeech('This is very good!'), 'GOOD', 'Good mapping');
assertEqual(getPrimaryActionFromSpeech('Please stop right there'), 'PLEASE', 'Please takes priority or stop detected');
assertEqual(getPrimaryActionFromSpeech('Stop now'), 'STOP', 'Stop mapping');

// Test 10: Sequential sentence extraction in chronological order
const seq = matchSpeechSequence('Hello, please help with water');
assertEqual(seq, ['HELLO', 'PLEASE', 'HELP', 'WATER'], 'Chronological sentence gesture sequence');

console.log('✅ ALL 25-CONCEPT PHRASE MATCHER TESTS PASSED!\n');
