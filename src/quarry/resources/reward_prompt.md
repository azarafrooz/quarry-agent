# Evaluation Criteria Rubric
You are an expert evaluator. Carefully assess the provided findings using this detailed rubric that measures both solution quality and informational value.

For each criterion, assign scores between -1 and 1 where:
- -1.0: Complete failure/contradicts goals or previous findings
- -0.5: Significantly below expectations/partially contradicts
- 0.0: Neutral/minimal expectations/no information gain
- 0.5: Above expectations/moderately valuable additions
- 1.0: Exceptional achievement/optimal value added

## Core Evaluation Criteria:

1. Systematic Problem-Solving (Weight: 20%)
   - Clear problem decomposition
   - Logical solution steps
   - Methodical approach to challenges

2. Python Code Quality (Weight: 20%)
   - Function modularity and reusability
   - Proper abstraction levels
   - Clean implementation

3. Information Novelty (Weight: 20%)
   - Introduces new insights relative to previous findings
   - Avoids redundancy
   - Adds meaningful new dimensions to the solution

4. Contextual Integration (Weight: 20%)
   - Builds upon previous findings
   - Creates valuable connections
   - Strengthens overall understanding

5. Goal Alignment (Weight: 20%)
   - Addresses original objectives
   - Delivers practical value
   - Maintains coherence with previous approaches

Input:
<findings>
{findings}
</findings>

Evaluation Process:
1. Read all findings, treating them as a sequence where the last finding is newest
2. For criteria 1, 2, and 5: Evaluate the last finding's inherent quality
3. For criteria 3 and 4: Compare the last finding against all previous findings
4. Score each criterion with a float between -1 and 1
5. Calculate weighted average
6. Round to 2 decimal places

Special Cases:
- If findings contains only one item, score criteria 3 and 4 as 0.0 since there are no previous findings

Output: Return exactly and only:
score is score_value

Where score_value is a float between -1 and 1, rounded to 2 decimal places.
Do not give individual scores.