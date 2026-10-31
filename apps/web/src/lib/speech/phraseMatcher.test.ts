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

console.log('--- Testing Phrase Matcher (ASL Vocabulary & Grammar) ---');

// Test 1: "thank you so much" should map to THANK_YOU with high priority
const m1 = getPrimaryActionFromSpeech('Thank you so much for your assistance');
assertEqual(m1, 'THANK_YOU', 'Multi-word Thank You mapping');

// Test 2: "sorry" should map to SORRY
const m2 = getPrimaryActionFromSpeech("I'm so sorry about that");
assertEqual(m2, 'SORRY', 'Apology phrase mapping');

// Test 3: "good morning" should map to HELLO / greeting
const m3 = getPrimaryActionFromSpeech('Good morning everyone!');
assertEqual(m3, 'HELLO', 'Greeting mapping');

// Test 4: "how are you" should map to HOW_ARE_YOU
const m4 = getPrimaryActionFromSpeech('How are you doing today?');
assertEqual(m4, 'HOW_ARE_YOU', 'How are you phrase mapping');

// Test 5: "namaste" should map to HELLO
const m5 = getPrimaryActionFromSpeech('Namaste, how are you?');
assertEqual(m5, 'HOW_ARE_YOU', 'Question takes conversational priority');

// Test 6: Word boundary protection — word like "bathroom" vs "restroom" (no false positives)
const m6 = getPrimaryActionFromSpeech('The telescope observed the galaxy');
assertEqual(m6, null, 'No false positives on random prose');

// Test 7: Multiple phrase prioritization in a single sentence
const m7 = matchSpeechPhrases('Hello and thank you so much');
assertEqual(m7[0].canonicalAction, 'THANK_YOU', 'Priority: multi-word thank you evaluated before single hello');
assertEqual(m7[1].canonicalAction, 'HELLO', 'Second action recognized in queue');

// Test 8: ASL Core concepts: HOSPITAL, BATHROOM, POLICE, EMERGENCY, GOOD, STOP
assertEqual(getPrimaryActionFromSpeech('Where is the nearest hospital?'), 'HOSPITAL', 'Hospital mapping');
assertEqual(getPrimaryActionFromSpeech('Where is the washroom?'), 'BATHROOM', 'Bathroom mapping');
assertEqual(getPrimaryActionFromSpeech('Call the police right now!'), 'POLICE', 'Police mapping');
assertEqual(getPrimaryActionFromSpeech('We have an emergency situation!'), 'EMERGENCY', 'Emergency mapping');
assertEqual(getPrimaryActionFromSpeech('This is very good!'), 'GOOD', 'Good mapping');
assertEqual(getPrimaryActionFromSpeech('Please stop right there'), 'PLEASE', 'Please takes priority or stop detected');
assertEqual(getPrimaryActionFromSpeech('Stop now'), 'STOP', 'Stop mapping');

// Test 9: Sequential sentence extraction with ASL Topic-Comment order
const seq = matchSpeechSequence('Where is the bathroom?');
assertEqual(seq, ['BATHROOM', 'WHERE'], 'Topic-Comment sentence gesture sequence');

console.log('✅ ALL ASL PHRASE MATCHER TESTS PASSED!\n');
