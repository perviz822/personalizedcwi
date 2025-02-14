<h1>Personalized CWI dataset


| word      | complex_percentage | length | frequency | aoa  | Concreteness | Familiarity |
| --------- | ------------------ | ------ | --------- | ---- | ------------ | ----------- |
| aerated   | 10.0               | 8      | 2.32      |      | 0.0          | 0.0         |
| abandoned | 37.5               | 9      | 4.4       |      | 0.0          | 0.0         |
| abdomen   | 10.0               | 7      | 3.56      | 8.61 | 586.0        | 426.0       |
| abducted  | 90.0               | 8      | 3.56      |      | 0.0          | 0.0         |
| abducting | 35.0               | 9      | 2.55      |      | 0.0          | 0.0         |
| abilities | 10.0               | 9      | 4.21      |      | 0.0          | 0.0         |

**word** - the annotated phrase

**complex_percentage** - the percentage  of non-native speakers (out of ten ) marking the word as complex

**length**- number of characters in the word

**frequency**- for obtaining the frequency of the word  wordfreq library is used. The **`wordfreq`** library provides word frequency data for over 40 languages, sourced from diverse corpora like Wikipedia, Twitter, and Google Books. The `zipf_frequency` function returns a word’s frequency on the **Zipf scale**, a logarithmic measure where higher values indicate more frequent usage. Here’s a detailed breakdown:

* **Range**: Typically `0.0` (extremely rare) to `8.5` (extremely common).
* **Interpretation**:
  * **7.0+**: Common words (e.g., "the" ≈ 7.3, "and" ≈ 7.1).
  * **6.0**: \~1 occurrence per million words (e.g., "flexed" ≈ 6.2).
  * **4.0**: \~1 occurrence per 100 million words (e.g., "glyph" ≈ 4.5).
  * **0.0**: Words not in the dataset (or below the threshold).

aoa- Age of acquisition, the source of this feature is from the kuperman dataset,basically  for each phrase people are asked  at which age do you think you  learned this phrase, and at the end the results are averaged for a word

**Concreteness and Familiarity**  i have obtained these features from **MRC Psycholinguistic Database**,The **MRC Psycholinguistic Database** provides **concreteness** and **familiarity** scores for English words, which are critical for modeling word complexity in tasks like Complex Word Identification (CWI). Here's a detailed breakdown of these features based on the research paper and MRC documentation:

**Concreteness**

* **Definition**: How tangible or abstract a word is (e.g., "apple" = concrete, "justice" = abstract).
* **Scale**: Scores range from **100–700** (integer values).
  * Higher values = more concrete (e.g., 670 for "chair").
  * Lower values = more abstract (e.g., 158 for "theory").

**Familiarity**

* **Definition**: How well-known a word feels to a reader.
* **Scale**: Scores range from **100–700**.
  * Higher values = more familiar (e.g., 657 for "apple").
  * Lower values = less familiar (e.g., 142 for "glyph").

For these features 0 indicates a missing value.
