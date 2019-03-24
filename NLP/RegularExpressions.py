# Import the regex module
import re

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.,?,!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))

####################################### Tokenization ########################################
# Import necessary modules
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)
#######################################
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"ARTHUR:"
print(re.match(pattern2, sentences[3]))

#################################################
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize


my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
pattern1 = r'\w+(\?!)'
pattern2 = r'(\w+|#\d|\?|!)'
pattern3 = r"(#\d\w+\?!)"
pattern4 = r"\s+"

print(pattern1, pattern2, pattern3, pattern4)
tokens = regexp_tokenize(my_string, pattern3)
print(tokens)
