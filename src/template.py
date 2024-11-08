

def get_cur_template(prompt_class, prompt_version):
    prompt_class_map = dict(
        only_q=GetCoTReasonProcess_TwoClue_only_Q,
        both_qa=GetCoTReasonProcess_TwoClue_both_QA,
        verify_by_a=VerifyCoTReasonProcess_by_Answer,
        optimize_prompt=OptimizePrompt,
        refine_prompt=RefineCoTPrompt,
    )
    return prompt_class_map[prompt_class].version2template[prompt_version]


    

class RefineCoTPrompt:
   version_list = ['v1030']
   model_call = 'o1-mini-2024-09-12'
   input_features = {
      'v1030': {'question', 'cot_result'},
   }
   version2template = dict(
v1030 = 
"""
以下给出了一个问题和一个推理过程，其中或许会有一些说得不够清楚或者不够准确的地方，甚至可能存在一些错误。请你帮忙对这个推理过程进行优化，使其更加清晰易懂、准确无误。

### 提供的问题以及答案
##### 问题
{question}
##### 答案
{cot_result}


### 输出格式(用你的回复替换"___")
```
___
```

请开始分析。

### 注意
1. 请判断提供的问题和答案是什么语种，然后你的输出也要用对应的语种输出。PS：如果是外国学生，请自行修改格式中的中文部分。

""",
)

class GetCoTReasonProcess_TwoClue_only_Q:
    version_list = ['v1017', 'v1018', 'v1021_top1']
    model_call = 'gpt-4o-2024-08-06'
    input_features = {
        'v1017': {'question'},
        'v1018': {'question'},
    }
    
    version2template = {'v1017':
"""You are an AI expert analyzing the image and the question provided. Follow these steps to ensure a thorough analysis:

# Note: You can swap steps 1 and 2 based on the question.

1. **Contextual Clue:** Extract and summarize any relevant contextual information from the text or question.
2. **Visual Clue:** Provide a detailed description of the elements and features present in the image.
3. **Step-by-Step Reasoning:** Use the contextual and visual clues to logically reason through the problem, considering each clue in detail.
4. **Final Answer:** Give the final answer based on the step-by-step reasoning.

### The Given Image and Question
{question}


Please proceed with the analysis.""",

# I add a 'Warning:'
'v1017_multilingual':
"""You are an AI expert analyzing the image and the question provided. Follow these steps to ensure a thorough analysis:

1. **Contextual Clue:** Extract and summarize any relevant contextual information from the text or question.
2. **Visual Clue:** Provide a detailed description of the elements and features present in the image.
3. **Step-by-Step Reasoning:** Use the contextual and visual clues to logically reason through the problem, considering each clue in detail.
4. **Final Answer:** Give the final answer based on the step-by-step reasoning.

### The Given Image and Question
{question}

Warning: Whatever language the question or the context of the image is in, your whole response should be in the same language!!
For example, if the question is in Chinese, your whole response including the reasoning process should be in Chinese, and vice versa.

Please proceed with the analysis.
""",

'v1017_zh':
"""
作为一个清华大学的优秀学生，你擅长回答各类问题。现在请您分析给定图片和问题。请按照以下步骤确保分析的全面性：

1. **上下文线索：** 从输入文本中提取并总结任何相关的上下文信息。
2. **视觉线索：** 对图片中存在的元素和特征进行详细描述。
3. **逐步推理：** 使用上下文和视觉线索进行逻辑推理，详细考虑每个线索。
4. **最终答案：** 根据逐步推理给出最终答案。

### 提供的图片，问题以及答案
{question}

请开始分析。
""",

# 比较难的再用这个v1021_top1，简单的用v1017_multilingual即可
'v1021_top1': 
"""You are an expert AI specializing in solving science problems with precision and accuracy. To ensure each solution is comprehensive, aligns correctly with the provided multiple-choice options, and addresses potential discrepancies, meticulously follow these enhanced structured steps:

1. **Problem Comprehension:**
   - **Read Carefully:** Thoroughly read the entire question to understand what is being asked.
   - **Identify Objectives:** Clearly state the primary objective (e.g., finding area, perimeter, angles).
   - **Highlight Key Information:** Underline or note all given measurements, geometric figures, and specific requirements.
   - **Restate the Problem:** Paraphrase the question in your own words to confirm understanding.

2. **Image and Diagram Analysis:**
   - **Examine the Diagram:** Analyze the provided image or diagram, noting all labels, angles, sides, and shapes.
   - **Identify Geometric Elements:** List all geometric elements present (e.g., triangles, parallelograms, circles).
   - **Determine Relationships:** Ascertain relationships between different geometric elements (e.g., parallel sides, congruent angles).
   - **Mark Critical Features:** Highlight any given lengths, angles, or other pertinent details that will aid in solving the problem.

3. **Extract and Organize Given Information:**
   - **List Numerical Values:** Enumerate all provided numerical measurements with their units.
   - **State Geometric Properties:** Identify any stated or implied geometric properties (e.g., congruence, similarity, right angles).
   - **Document Assumptions:** Clearly specify any assumptions based on the problem's context and ensure they are justified.

4. **Select Appropriate Geometric Principles and Formulas:**
   - **Determine Relevant Theorems/Formulas:** Identify which geometric theorems or formulas apply to this problem (e.g., Pythagorean theorem, area formulas, trigonometric ratios).
   - **Justify Selection:** Explain why the chosen principles or formulas are relevant to solving the problem.

5. **Detailed Step-by-Step Reasoning:**
   - **Step 1:** Apply the selected theorem or formula to set up the initial equation or expression.
   - **Step 2:** Substitute the known values into the equation accurately.
   - **Step 3:** Perform calculations systematically, ensuring each step follows logically from the previous one.
   - **Step 4:** Use the correct mathematical operations and trigonometric functions, verifying their appropriate application based on the geometric context.
   - **Step 5:** Solve for the unknown variable, detailing each calculation to prevent errors.

6. **Solution Verification:**
   - **Review Calculations:** Double-check each calculation step for mathematical accuracy.
   - **Validate Units and Dimensions:** Ensure all units are consistent and calculations make sense within the geometric context.
   - **Confirm Logical Consistency:** Verify that each step follows logically and that no assumptions contradict the given information.
   - **Cross-Check with Diagram:** Refer back to the diagram to ensure all elements have been accurately considered and utilized.

7. **Answer Alignment and Multiple-Choice Selection:**
   - **Calculate Final Answer:** Determine the numerical value of the unknown, rounding as necessary per the problem's instructions.
   - **Compare with Choices:** Match your calculated answer with the provided multiple-choice options.
   - **Direct Match:**
     - If the calculated answer exactly matches one of the options, select that option confidently.
   - **Format Adjustment:** 
     - If the calculated answer is equivalent but formatted differently (e.g., fraction vs. decimal, simplified vs. unsimplified), adjust the format to match the options before selecting.
   - **Non-Matching Answer:**
     - **Re-evaluate Calculations:** Systematically revisit each step to identify and correct any possible mistakes.
     - **Reassess Assumptions:** Check if any incorrect assumptions were made that could have led to the discrepancy.
     - **Consider Alternative Approaches:** Explore different methods or formulas that might apply to the problem.
     - **Determine Approximation Necessity:** Assess if rounding or approximation is appropriate based on the problem context.
     - **Select the Closest Option:** If no exact match is found after thorough re-evaluation, choose the option closest to your calculated value.
     - **Document Reasoning:** Clearly explain why the closest option was selected despite the discrepancy, outlining any adjustments or considerations made.

8. **Final Answer Presentation:**
   - **State the Correct Option:** Clearly provide the correct option letter along with its corresponding value.
   - **Provide Comprehensive Justification:** Elaborate on why this option is correct, referencing your calculations, alignment with the problem requirements, and consistency with the provided options.
   - **Summarize Key Points:** Briefly recap the critical steps that led to the correct answer, reinforcing the logical flow of your reasoning.

### The Given Image and Question
{question}

Please proceed with the detailed analysis.""",

'ai2d' : 
"""You are an expert AI specializing in solving scientific multiple-choice questions with precision and accuracy. To ensure each solution is comprehensive, aligns correctly with the provided options, and eliminates ambiguity, meticulously follow these enhanced structured steps:

1. **Problem Comprehension:**
   - **Read Carefully:** Thoroughly read the entire question, including any accompanying images or diagrams, to understand what is being asked.
   - **Identify Objectives:** Clearly state the primary objective (e.g., identifying the most affected species, understanding ecological relationships).
   - **Highlight Key Information:** Underline or note all important terms, concepts, and constraints mentioned in the question and options.
   - **Restate the Problem:** Paraphrase the question in your own words to confirm understanding.

2. **Contextual Knowledge Application:**
   - **Recall Relevant Information:** Utilize your knowledge base to recall facts, theories, and principles related to the question topic.
   - **Identify Key Concepts:** Determine the key scientific concepts involved (e.g., life cycle stages, ecological dependencies, biological interactions).
   - **Establish Relationships:** Analyze how the concepts interact within the context of the question (e.g., predator-prey relationships, life cycle progression).

3. **Image and Diagram Analysis (if applicable):**
   - **Describe Visual Elements:** If an image is provided, describe its key components relevant to the question.
   - **Interpret Visual Information:** Relate the visual elements to the question’s context using scientific principles.

4. **Option Analysis:**
   - **List All Options:** Enumerate all provided multiple-choice options.
   - **Evaluate Each Option:**
     - **Relevance:** Assess how each option relates to the key concepts identified.
     - **Accuracy:** Ensure each option precisely fits within the scientific framework relevant to the question.
     - **Support with Evidence:** Reference specific facts or principles that support or refute each option's suitability.
   - **Eliminate Incorrect Options:** Systematically remove options that are clearly unrelated, irrelevant, or factually incorrect based on the evaluation.

5. **Detailed Reasoning Process:**
   - **Step 1:** Identify the primary factor or concept central to the question (e.g., type of metamorphosis: complete or incomplete).
   - **Step 2:** Analyze how each remaining option aligns with this primary factor.
   - **Step 3:** Use logical deduction to determine which option most directly and significantly satisfies the question’s requirement.
   - **Step 4:** Ensure that the reasoning aligns with established scientific knowledge and principles without presenting multiple conflicting conclusions.

6. **Solution Verification:**
   - **Review Reasoning:** Double-check each reasoning step for logical consistency and factual accuracy.
   - **Cross-Reference Knowledge:** Ensure that the conclusion aligns with known scientific facts and relationships.
   - **Confirm Single Answer Selection:** Validate that only one option is selected as the most appropriate based on the analysis.

7. **Final Answer Selection:**
   - **Determine Correct Option:** Based on the thorough analysis, select the single option that best fits the question’s requirements.
   - **Provide Justification:** Clearly and concisely explain why this option is the most appropriate, referencing the key concepts and reasoning steps used.
   - **Summarize Key Points:** Briefly recap the critical factors that led to the correct answer, reinforcing the logical flow of your reasoning.

### The Given Image and Question
{question}

Please proceed with the detailed analysis.""",

'raven': """For each Raven's Progressive Matrices problem, follow the comprehensive steps below to accurately solve the puzzle:

1. **Comprehend the Problem:**
   - **Examine the Grid Thoroughly:** Observe the entire grid of patterns/images meticulously. Note the number of rows and columns, and provide a detailed description of each existing image, including shapes, colors, orientations, sizes, and shading.
   - **Identify Missing Elements:** Locate any missing pieces or placeholders within the grid that require completion.

2. **Identify and Catalog Patterns:**
   - **Detect Recurring Themes:** Look for repeating shapes, colors, numbers, orientations, sizes, and shading across all rows and columns.
   - **Analyze Transformations:** Observe how elements transform from one cell to another, such as through progression, rotation, reflection, scaling, or other geometric transformations.
   - **Highlight Regularities:** Document any consistent trends or rules that apply both horizontally and vertically within the grid.

3. **Develop and Validate Hypotheses:**
   - **Formulate Potential Rules:** Based on the identified patterns, propose possible rules that govern the sequence and arrangement of images.
   - **Ensure Comprehensive Coverage:** Verify that these rules consistently apply across all relevant rows and columns, leaving no exceptions unaccounted for.
   - **Consider Alternative Explanations:** Explore other plausible patterns or rules that might also explain the observed arrangements to ensure robustness.

4. **Systematically Evaluate Answer Choices:**
   - **Review All Options Carefully:** Examine each provided option in detail, noting how each aligns with the identified patterns and rules.
   - **Match Options Against Patterns:**
     - **Comprehensive Comparison:** Compare each option against every established pattern and rule, ensuring alignment on all identified aspects.
     - **Consistency Verification:** Assess whether the option maintains consistency across both horizontal and vertical trends.
   - **Eliminate Inconsistent Choices:** Disregard any options that fail to satisfy one or more of the established patterns or rules.

5. **Draw and Justify a Conclusion:**
   - **Determine the Best Fit:** Identify the option that most comprehensively and consistently aligns with all identified patterns and rules.
   - **Articulate Clear Justification:** Provide a concise explanation of how the selected option fulfills each relevant pattern and rule, reinforcing its correctness.
   - **Address Potential Alternatives:** Briefly explain why other options were eliminated based on inconsistencies with the patterns and rules.

6. **Finalize and Present the Answer:**
   - **State the Final Answer Clearly:** Explicitly provide the chosen option as the answer. If a "Yes/No" response is required, ensure the answer is unambiguously one of these.
   - **Provide a Concise Justification:** Summarize the reasoning that led to the selected answer, directly linking it to the identified patterns and rules.

7. **Review and Validate the Reasoning:**
   - **Cross-Check All Steps:** Re-examine each part of the reasoning process to ensure logic flow and the absence of errors.
   - **Confirm Alignment with Patterns:** Ensure that the final answer aligns perfectly with all the identified patterns and rules without exception.
   - **Ensure Answer Format Compliance:** Verify that the answer adheres strictly to the required format, whether it involves selecting an option or providing a "Yes/No" response.
   
### The Given Image and Question
{question}

Please proceed with the detailed analysis."""
}
    
    

class GetCoTReasonProcess_TwoClue_both_QA:
    version_list = ['v1017', 'v1018']
    model_call = 'gpt-4o-2024-08-06'
    input_features = {
        'v1017': {'question', 'answer'},
        'v1018': {'question', 'answer'},
    }
    
    version2template = {

'v1105_ft_bak1':
"""You are an AI expert analyzing the image, the question, and the provided answer. Your task is to provide a detailed and thorough analysis, ensuring that the reasoning process is clear and logical. Follow these steps:

1. **Contextual Clue:** Extract and summarize any relevant contextual information from the text or question.
2. **Visual Clue:** Provide a detailed description of the elements and features present in the image.
3. **Step-by-Step Reasoning:** Use the contextual and visual clues to logically reason through the problem, considering each clue in detail. 
4. **Final Answer:** Confirm the provided answer and explain why it is correct based on the step-by-step reasoning.

### Given Information
<image>
Question: {question}
Answer: {answer}

Warning: 
1. Whatever language the question or the context of the image is in, your whole response should be in the same language!! For example, if the question is in Chinese, your whole response, including the reasoning process, should be in Chinese, and vice versa.
2. Ensure that your reasoning process is comprehensive and detailed, without directly revealing that you have seen the provided answer.
3. If the question is complex, follow the above steps for a detailed and thorough analysis. If the question is straightforward and the answer is immediately obvious, follow this format:
   - **Observation and Analysis:** [A brief observation and analysis in 1-2 sentences]
   - **Conclusion:** [The concise answer in 1 sentence]

Please proceed with the analysis, providing a detailed and thorough explanation of each step.
""",

'v1105_ft':
"""You are an AI expert analyzing the image, the question, and the provided answer. Your task is to provide a detailed and thorough analysis, ensuring that the reasoning process is clear and logical.

### Important Note:
If the question is complex, follow the above steps for a detailed and thorough analysis. A complex question for a visual large model (VLM) typically involves multiple steps of reasoning, requires understanding of context, or involves intricate visual elements. The output format for a complex question should be:
   - **Contextual Clue:** [Extract and summarize any relevant contextual information from the text or question]
   - **Visual Clue:** [Provide a detailed description of the elements and features present in the image]
   - **Step-by-Step Reasoning:** [Use the contextual and visual clues to logically reason through the problem, considering each clue in detail]
   - **Final Answer:** [Confirm the provided answer and explain why it is correct based on the step-by-step reasoning]

If the question is straightforward and the answer is immediately obvious, follow this format:
   - **Initial Observation and Analysis:** [A brief observation and analysis in 1-2 sentences]
   - **Conclusion:** [The concise answer in 1 sentence]
   A straightforward question for a VLM usually involves simple visual recognition or a direct interpretation of the image content.

### Given Information
<image>
Question: {question}
Answer: {answer}

### Warning: 
1. Whatever language the question or the context of the image is in, your whole response should be in the same language!! For example, if the question is in Chinese, your whole response, including the reasoning process, should be in Chinese, and vice versa.
2. Ensure that your reasoning process is comprehensive and detailed, without directly revealing that you have seen the provided answer.

Please proceed with the analysis, providing a detailed and thorough explanation of each step when necessary.
""",

'tiku1104': 
"""作为一位清华大学的优秀学生，你接受了一份家教的工作，你擅长给各类题目给出非常到位的解析，从而让学生轻松理解。
现在请您根据给定的题目及其原始的简洁答案，重新进行分析，给出"详细到位的解析过程"。
例如，对于给定的题目和原始简洁答案，你会给出包含以下内容的"详细到位的解析过程"：
1. **题目解析：** 仔细阅读题目，理解问题的关键点。
2. **知识点回忆：** 回忆相关知识点，准备解答问题。
3. **逐步推理：** 根据题目解析和知识点回忆，逐步推理出最终答案。
4. **最终答案：** 根据逐步推理给出最终答案。

现在将这个范式应用到以下材料中，并按照指定格式输出"详细到位的解析过程"。

### 给定的题目和原始简洁答案
问题: ```{question}```
答案: ```{answer}```

### 输出格式
##### 1. "思维链"格式
```markdown
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___
```

##### 2. "直接回答"格式
```markdown
### 回忆题目相关知识点
___

### 总结得到最终答案
___
```

### 注意
1. 请先判断给定题目的难度，选择合适的输出格式。
2. 如果题目较为简单，则按照"直接回答"格式输出。
3. 如果题目较为复杂，则按照"思维链"格式输出。
4. 格式中的"___"表示你的解答内容。输出时请替代掉"___"。
5. 你的输出是"详细到位的解析过程"，将直接配套作为题目的解析过程，给学生查阅。所以其中不应该带有prompt中相关的信息，例如不要涉及"直接回答格式" "提供的原始简洁"等字样，只需提供解析过程即可。

请开始分析。
""",

# 5. 严格按照格式输出。不要再带有任何关于选择哪种格式的输出！例如含有"直接回答"，"思维链"等的句子。而是直接按照"直接回答"格式和"思维链"格式输出。


'v1017':
"""I need help understanding the steps between an image, a question, and its answer. Please:
1. Visual Clues: Identify key elements in the image that are relevant.
2. Contextual Clues: Explain any important background information or context.
3. Reasoning Process: Provide a step-by-step reasoning to reach the answer.

Question: {question}
Answer: {answer}""",


'v1017_zh':
"""作为一位清华大学的优秀学生，你接受了一份家教的工作，你擅长给各类题目给出非常到位的解析，从而让学生轻松理解。
现在请您分析给定的题目，为这个问题提供详细的解答过程。

1. **题目解析：** 仔细阅读题目，理解问题的关键点。
2. **知识点回忆：** 回忆相关知识点，准备解答问题。
3. **逐步推理：** 根据题目解析和知识点回忆，逐步推理出最终答案。
4. **最终答案：** 根据逐步推理给出最终答案。

### 提供的图片，问题以及答案
{question}
{answer}

### 输出格式
```markdown
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___
```

请开始分析。

1. 请判断传入图片的学生是中国学生还是外国学生，然后你的输出也要用对应的语种输出。PS：如果是外国学生，请自行修改格式中的中文部分。
""",


'v1017_code':
"""
作为一名在LeetCode社区活跃多年的编程高手，您不仅拥有丰富的编程经验和深厚的算法知识，还擅长将复杂的概念用简单易懂的方式解释清楚。您乐于助人，经常在社区里分享解题思路和编程技巧，帮助许多初学者和进阶者提升了他们的编程水平。现在，您收到了一位编程爱好者的请求，需要针对他们遇到的LeetCode题目进行深入浅出的解析，以帮助他们更好地掌握解题思路和技巧。

### 0. 任务
请根据给定的LeetCode题目及其图片，提供一个详细而清晰的解答过程，帮助他们理解问题并掌握解题方法。


### 1. 提供的问题以及答案
##### 问题
{question}
##### 答案
'''
```
{answer}
```
'''

### 2. 输出示例
```markdown
## 文字题解

#### 思路

我们可以如下递归地定义两个链表里的 `merge` 操作（忽略边界情况，比如空链表等）：

\[ \\text{{list1}}[0] + \\text{{merge}}(\\text{{list1}}[1:], \\text{{list2}}) \quad \\text{{list1}}[0] < \\text{{list2}}[0] \]

\[ \\text{{list2}}[0] + \\text{{merge}}(\\text{{list1}}, \\text{{list2}}[1:]) \quad \\text{{otherwise}} \]


也就是说，两个链表头部值较小的一个节点值与其余元素的 `merge` 操作结果合并。

#### 算法

我们直接将以上递归过程建模， 同时需要考虑边界情况。

如果 `l1` 或者 `l2` 一开始就是空链表的话，那么没有任何操作需要合并，所以我们直接返回非空链表。否则，我们要判断 `l1` 和 `l2` 哪一个链表头部的元素更小，然后递归地决定下一个添加到结果里的节点。如果两个链表有一个为空，递归结束。

```java
class Solution {{
    // 这个函数合并两个有序的链表，并返回合并后的有序链表的头节点
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {{
        // 如果第一个链表为空，返回第二个链表
        if (l1 == null) {{
            return l2;
        // 如果第二个链表为空，返回第一个链表
        }} else if (l2 == null) {{
            return l1;
        }}

        // 比较两个节点的值
        // 如果第一个链表的节点值较小
        // 将第一个链表的下一个节点设置为递归调用结果，传入第一个链表的下一个节点和当前的第二个链表节点
        // 返回当前的第一个链表节点
        }} else if (l1.val < l2.val) {{
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        // 如果第二个链表的节点值较小或相等
        // 将第二个链表的下一个节点设置为递归调用结果，传入当前的第一个链表节点和第二个链表的下一个节点
        // 返回当前的第二个链表节点
        }} else {{
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }}
    }}
}}
```

#### 复杂度分析

- 时间复杂度：\( O(n + m) \)，其中 \( n \) 和 \( m \) 分别为两个链表的长度。因为每次递归调用都会去掉 `l1` 或者 `l2` 的头节点（直到至少有一个链表为空），函数 `mergeTwoLists` 至多只会递归调用每个节点一次。因此，时间复杂度取决于合并后的链表长度，即 \( O(n + m) \)。

- 空间复杂度：\( O(n + m) \)，其中 \( n \) 和 \( m \) 分别为两个链表的长度。递归调用 `mergeTwoLists` 函数时需要消耗栈空间， 栈空间的大小取决于递归调用的深度。 结点递归调用时 `mergeTwoLists` 函数最多调用 \( n + m \) 次，因此空间复杂度为 \( O(n + m) \)。
```


## 3. 输出要求
i. 一定要以```markdown\n___```格式输出，注意用你的回复替换掉"___"。
""",


'v1017_zh_multi_ques':
"""作为一位清华大学的优秀学生，你接受了一份家教的工作，你擅长给各类题目给出非常到位的解析，从而让学生轻松理解。
现在请您分析给定的题目，为这个问题提供详细的解答过程。

1. **题目解析：** 仔细阅读题目，理解问题的关键点。
2. **知识点回忆：** 回忆相关知识点，准备解答问题。
3. **逐步推理：** 根据题目解析和知识点回忆，逐步推理出最终答案。
4. **最终答案：** 根据逐步推理给出最终答案。

### 提供的图片，问题以及答案
{question}
{answer}


### 输出格式
#### 1. 图片中仅含一道小题
```markdown
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___
```

#### 2. 图片中含有多道小题
```markdown
该图片中含有多道小题，我将分小题逐一解答。

## 第一道小题
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___

## 第二道小题
...(同上，依次类推)
```

### 注意
1. 如果题目中有多个小题，请你现在输出一行："该图片中含有多道小题，我将分小题逐一解答。"，然后再针对每个小问题，各自输出一段输出格式中要求的markdown格式的解答，如4.2所示。"___"表示你的解答内容。输出时请替代掉"___"。
2. 如果题目中只有一道小题，按照4.1所示的格式输出即可。

请开始分析。
""",



'v1017_zh_multi_ques_tune':
"""作为一位清华大学的优秀学生，你接受了一份家教的工作，你擅长给各类题目给出非常到位的解析，从而让学生轻松理解。
现在请您分析给定的题目，为这个问题提供详细的解答过程。

### 解答过程
1. **题目解析：** 仔细阅读题目，理解问题的关键点。
2. **知识点回忆：** 回忆相关知识点，准备解答问题。
3. **逐步推理：** 根据题目解析和知识点回忆，逐步推理。
4. **最终答案：** 根据逐步推理给出最终答案。

### 题目及其答案
##### 题目
{question}
##### 答案
{answer}

### 输出格式
#### 格式A. "only one question"的输出格式
```markdown
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___
```

#### 格式B. "contains multiple sub-questions, each independently numbered"的输出格式
```markdown
该图片中含有多道小题，我将分小题逐一解答。

## 第一道小题
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___

## 第二道小题
...(同上，依次类推，直至最后一道小题，每道小题的格式相同)
```

### 注意
1. 请先判断给定题目是 "only one question"(例如一道选择题，一道问答题)，还是 "contains multiple sub-questions, each independently numbered"。进而正确选择对应的输出格式。
2. 如果是 "only one question"，请以下述的"格式A"格式输出。
3. 如果是 "contains multiple sub-questions, each independently numbered"，请以下述的"格式B"格式输出。
4. 格式中的"___"表示你的解答内容。输出时请替代掉"___"。
5. 如果图片中虽然有第二个小题，但是没有拍清楚，那么将该问题视为"only one question"的情况。
6. 你的输出中，不要带有任何关于思考选择哪种格式的句子，例如"only one question"，"contains multiple sub-questions, each independently numbered"，"格式A"，"格式B"等等。

请开始分析。
""",

'v1017_zh_ans_img':
"""作为一位清华大学的优秀学生，你接受了一份家教的工作，你擅长给各类题目给出非常到位的解析，从而让学生轻松理解。
现在请您分析给定的题目，为这个问题提供详细的解答过程。

1. **题目解析：** 仔细阅读题目，理解问题的关键点。
2. **知识点回忆：** 回忆相关知识点，准备解答问题。
3. **逐步推理：** 根据题目解析和知识点回忆，逐步推理出最终答案。
4. **最终答案：** 根据逐步推理给出最终答案。

### 提供的图片，问题以及答案

#### 问题和问题所含图片
{question}

#### 答案和答案所含图片
{answer}


### 输出格式
#### 1. 图片中仅含一道小题
```markdown
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___
```

#### 2. 图片中含有多道小题
```markdown
该图片中含有多道小题，我将分小题逐一解答。

## 第一道小题
### 题目解析
___

### 知识点回忆
___

### 逐步推理
___

### 最终答案
___

## 第二道小题
...(同上，依次类推)
```

### 注意
1. 如果题目中有多个小题，请你首先输出一行："该图片中含有多道小题，我将分小题逐一解答。"，然后再针对每个小问题，各自输出一段输出格式中要求的markdown格式的解答，如4.2所示。"___"表示你的解答内容。输出时请替代掉"___"。
2. 如果题目中只有一道小题，按照4.1所示的格式输出即可。
3. 如果传入了两张图片，请你根据prompt中<image>的位置，判断是否是答案中含有图片。请你在逐步推理中加入答案图片的描述！
如果是，请你首先输出一行："需要作图解析，但我会在答案解析中加入图片描述。"，然后再按照4.1或4.2所示的格式输出。


请开始分析。
""",


}
    
    
class VerifyCoTReasonProcess_by_Answer:
    version_list = ['v1017', 'v1018']
    model_call = 'gpt-4'
    input_features = {
        'v1017': {'question', 'answer'},
        'v1018': {'question', 'answer'},
    }
    
    version2template = {'v1017':
"""### Question
'''{question}'''

### Right Answer
'''{answer}'''

### Student's "Reasoning Process and Conclusion"
'''{cot_result}'''

### Task
You're judging whether the student's "Reasoning Process and Conclusion" is correct.

### Output Format
'''
1. Output: x
2. Reason: ___
'''
x should be one of the following: True, False, NoAnswer.

Please stick to the format.
"""}
    

class OptimizePrompt:
    version_list = ['v1017', 'v1018']
    model_call = 'gpt-4o-2024-08-06'
    input_features = {
        'v1017': {'task_description', 'history', 'error_analysis', 'labels'},
        'v1018': {'task_description', 'history', 'error_analysis', 'labels'},
    }
    # 
    version2template = {
'v1017':

"""You are a proficient prompt engineer specializing in large language model development. You excel at analyzing prompts from batch inferences to identify their shortcomings, and subsequently readjusting and optimizing the prompts to make them more accurate and effective. 

Currently, we are focused on improving the performance of large language models in solving geometric math problems. Given the following CoT (Chain-of-Thought) prompt and its corresponding error cases, please boldly rewrite the original CoT_prompt_template so that the new CoT_prompt_template can better address these error cases.

### 1. Original CoT_prompt_template
```{CoT_prompt_template}```

### 2. Error cases (including the {{question}}, incorrect reasoning process, summarized error reasons)
##### 2.1 Error case {{question}}, which will be incorporated into the original CoT_prompt_template to form a complete CoT_prompt.
```{question}```

##### 2.2 Error case reasoning process
```{cot_result}```

##### 2.3 Error case summarized reasons for error
```{verify_result}```


### 3. Task
Based on the above bad cases, boldly rewrite the original CoT_prompt_template to make it more accurate and effective in reasoning geometric math problems.


### 4. Notes
4.1. Do not change the variable placeholders like {{question}} in the original prompt template.


### 5. Output format
5.1. Output format: ```markdown\n{{new CoT_prompt_template}}```
5.2. Fill in the new CoT_prompt_template directly in the output format without any extra explanation.
""",

# 在3task那里多加了一句
'v1018':

"""You are a proficient prompt engineer specializing in large language model development. You excel at analyzing prompts from batch inferences to identify their shortcomings, and subsequently readjusting and optimizing the prompts to make them more accurate and effective. 

Currently, we are focused on improving the performance of large language models in solving science problems. Given the following CoT (Chain-of-Thought) prompt and its corresponding error cases, please boldly rewrite the original CoT_prompt_template so that the new CoT_prompt_template can better address these error cases.

### 1. Original CoT_prompt_template
```{CoT_prompt_template}```

### 2. Error cases (including the {{question}}, incorrect reasoning process, summarized error reasons)
##### 2.1 Error case {{question}}, which will be incorporated into the original CoT_prompt_template to form a complete CoT_prompt.
```{question}```

##### 2.2 Error case reasoning process
```{cot_result}```

##### 2.3 Error case summarized reasons for error
```{verify_result}```


### 3. Task
Based on the above bad cases, boldly rewrite the original CoT_prompt_template to make it more accurate and effective in reasoning science problems.
Remember to use all the prompt engineering skills you have to optimize the prompt, ensuring clarity, specificity, and a focus on actionable information while considering the context and desired outcome. Additionally, employ the Chain of Thought (CoT) methodology to break down complex tasks into manageable steps, enhancing both the effectiveness and the comprehensibility of the prompt.

### 4. Notes
4.1. Do not change the variable placeholders like {{question}} in the original prompt template.


### 5. Output format
5.1. Output format: ```markdown\n{{new CoT_prompt_template}}```
5.2. Fill in the new CoT_prompt_template directly in the output format without any extra explanation.
""",

'zh_v1017':
"""你是一位优秀的大模型方向的prompt工程师，你擅长分析prompt在批量推理时产出的，进而发现prompt中不足的地方，然后重新调整和优化提示语，使其更为准确和有效。
当前，我们正致力于提升大模型在几何数学题方面的表现。给定以下CoT（Chain-of-Thought）prompt及其对应的推理错误case，请你大胆改写原始的CoT_prompt_template，使新的CoT_prompt_template能够更好地解决这些错误case.

### 原始的CoT_prompt_template
```{CoT_prompt_template}```

### 错误case
```{question}{cot_result}```

### 任务
请基于上述badcase大胆改写原始的CoT_prompt_template，使其在推理几何数学题时更为准确和有效。

### 注意
1. 原始prompt中的变量占位符如{question}、{answer}等不需要改动，并且不能变少。

### 输出格式
1. 输出格式：```markdown\n{新的CoT_prompt_template}```
2. 直接将新的CoT_prompt_template填入输出格式中即可，无需额外解释。
"""
}
    


t = """You are a proficient prompt engineer specializing in large language model development. You excel at analyzing prompts from batch inferences to identify their shortcomings, and subsequently readjusting and optimizing the prompts to make them more accurate and effective. 

Currently, we are focused on improving the performance of large language models in solving science problems. Given the following CoT (Chain-of-Thought) prompt and its corresponding error cases, please boldly rewrite the original CoT_prompt_template so that the new CoT_prompt_template can better address these error cases.

### 1. Original CoT_prompt_template
```{CoT_prompt_template}```

### 2. Error cases (including the problem, incorrect reasoning process, summarized error reasons)
##### 2.1 Error case problem, which will be incorporated into the original CoT_prompt_template to form a complete CoT_prompt_template
```{question}```

##### 2.2 Error case reasoning process
```{cot_result}```

##### 2.3 Error case summarized reasons for error
```{verify_result}```


### 3. Task
Based on the above bad cases, boldly rewrite the original CoT_prompt_template to make it more accurate and effective in reasoning science problems.


### 4. Notes
4.1. Do not change the variable placeholders like {{question}} in the original prompt.


### 5. Output format
5.1. Output format: ```markdown\n{{new CoT_prompt_template}}```
5.2. Fill in the new CoT_prompt_template directly in the output format without any extra explanation.
"""

a = t.format(
    **{
    "cot_result":"1. **Contextual Clue:**\n   - The question asks for the area of a kite given specific measurements.\n   - The formula for the area of a kite is \\(\\text{Area} = \\frac{1}{2} \\times d_1 \\times d_2\\), where \\(d_1\\) and \\(d_2\\) are the lengths of the diagonals.\n\n2. **Visual Clue:**\n   - The given image shows a kite with diagonals labeled as 14 ft and 8 + 5 ft.\n   - The total length of the second diagonal is the sum of 8 ft and 5 ft, which is 13 ft.\n\n3. **Step-by-Step Reasoning:**\n   - Calculate the total length of the diagonals:\n     - First diagonal (\\(d_1\\)) = 14 ft.\n     - Second diagonal (\\(d_2\\)) = 8 ft + 5 ft = 13 ft.\n   - Apply the area formula for the kite:\n     - \\(\\text{Area} = \\frac{1}{2} \\times 14 \\times 13\\).\n     - Calculate: \\(\\frac{1}{2} \\times 14 \\times 13 = \\frac{1}{2} \\times 182 = 91\\).\n\n4. **Final Answer:**\n   - None of the provided choices directly match the calculated area (91 square feet), which suggests there might be an oversight. However, based on given options and standard process, the closest to 91 among typical problem-solving logic and choices provided could be a discrepancy, but the intended answer should still follow visual calculations showing 91 square feet.\n   \n   Therefore, based on direct calculations, proceed with verification or follow original process if different choices are confirmed differently, but calculations here point to 91.",
    "verify_status":200,
    "verify_result":"'''\n1. Output: False\n2. Reason: The student's calculation is incorrect. The correct area of the kite is calculated as \\(\\frac{1}{2} \\times 14 \\times 13 = 91\\). However, they concluded the answer as C: 110 which does not match their calculation of 91. The student also incorrectly questioned the correct area found without recognizing any possible error in provided choices or in their original reasoning, leading to an incorrect conclusion.\n'''",
    "verify_tag":"False",
    "question": "<image>Answer the mathemtical geometry problem and directly provide the correct option letter.\nQuestion: Find the area of the kite.\nChoices: A: 40 B: 70 C: 110 D: 220\nShort Answer.",
    "CoT_prompt_template": 
"""You are an AI expert analyzing the image and the question provided. Follow these steps to ensure a thorough analysis:

1. **Contextual Clue:** Extract and summarize any relevant contextual information from the text or question.
2. **Visual Clue:**  Provide a detailed description of the elements and features present in the image.
3. **Step-by-Step Reasoning:** Use the contextual and visual clues to logically reason through the problem, considering each clue in detail.
4. **Final Answer:** Give the final answer based on the step-by-step reasoning.

### The Given Image and Question
\{question\}


Please proceed with the analysis."""

    }
)



new2 = """
You are an AI expert specialized in analyzing images and solving science problems. Follow these steps meticulously to ensure accurate and effective problem-solving:

1. **Contextual Understanding:** Clearly understand the problem statement and what is being asked. Identify and list any given values, formulas, or relevant information from the text or question.
2. **Visual Analysis:** Examine the given image meticulously. Describe all relevant geometrical figures, labeling their dimensions or other critical features accurately.
3. **Accurate Calculation:** Use the information from the contextual understanding and visual analysis to perform precise mathematical calculations. Ensure all values are correctly identified and used in the proper geometric formulas.
4. **Step-by-Step Reasoning:** Clearly document each step of your reasoning, showing all intermediate calculations and logical steps to ensure transparency and accuracy.
5. **Verification:** Double-check your calculations and results against the given options, ensuring your final answer is logically consistent and correctly derived.
6. **Final Answer:** Provide the final answer based on your step-by-step reasoning and verification.

### The Given Image and Question
{question}

Please proceed with the analysis.
"""

new1 = """
You are an AI expert specialized in mathematical geometry analysis. Follow these steps to ensure a thorough and accurate solution:

1. **Problem Understanding:** Clearly identify what geometric shape is being discussed and what is being asked.
2. **Extract Information:** Summarize and extract all given measurements, constants, and relevant details provided in the text or question.
3. **Formula Selection:** Identify and write down the appropriate geometric formulas needed to solve the problem.
4. **Calculate Step-by-Step:** Perform each calculation carefully, breaking down the steps logically.
5. **Verification:** Check if the final calculation matches any of the provided choices. If there is a discrepancy, re-evaluate the given details and calculations to ensure no numerical or logical errors occurred.
6. **Final Answer:** Provide the answer that fully matches the correctly verified calculation, if none match, indicate that there is a mismatch.

### The Given Image and Question
{question}

Please proceed with the analysis.
"""