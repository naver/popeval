# popEval

This proposed evaluation algorithm is not only applicable to current end-to-end tasks, but also suggests a new direction to redesign the evaluation concept for further OCR researches.

Keywords: end-to-end evaluation; character level evaluation; character-oriented evaluation, optical character recognition;

## Usage

### Prediction and ground truth file format
8 coordinates( 4 points of (x,y) format ) and text.
- Both (x,y) and (y,x) are acceptable. But all data should be consistent.
- Order of points are not important. They will be reordered when evaluating.
- The splitter between coordinates and text is <code>##::</code> by default.
#### Example.
<pre>
2490 2774 2614 2769 2618 2869 2494 2875##::Thai
2636 2824 2863 2814 2865 2859 2638 2869##::Cuisihe
2701 2830 2761 2835 2758 2866 2698 2861##::11s
1423 3108 2147 3076 2159 3344 1435 3376##::Thai
2270 3070 3480 3017 3492 3291 2282 3344##::Cuisihe
</pre>

### Run on command prompt
<pre>$ python popEval.py --gtpath gt --dtpath dt</pre>
The script will match files of the same name between the two directories.
- Unmatched files will be ignored
- This script is intended to work regardless of the python version.(tested on 2.7 and 3.x)

# Dataset

ICDAR13 focused scene dataset's test data,
ICDAR15 incidental scene dataset's test data.

We newly annotated them at character level. The format is as above.
Note that un-annotated parts of the original data are not also at the character level.

# Citation
If you find this work useful for your research, please cite:

```
@inproceedings{popeval19,
  title={PopEval: A Character-Level Approach to End-To-End Evaluation Compatible with Word-Level Benchmark Dataset},
  author={Hong-Seok Lee, Youngmin Yoon, Pil-Hoon Jang, Chankyu Choi},
  booktitle = {International Conference of Document Analysis and Recognition (ICDAR)},
  year={2019}
}
```

# License

```
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
