Complete the following exercise to use scikit-learn
or parsnip to solve a classification problem, to
predict the type of chess pieces based on their base
diameter and height.

## scikit-learn

1. Copy the contents of `30_sklearn_regress.py` into
   a new file named `30_sklearn_classify.py`.

2. In section **0. Preliminaries**, add the following
   additional imports:
   ```python
   from sklearn.preprocessing import LabelEncoder
   from sklearn.tree import DecisionTreeClassifier
   ```

3. In section **2. Prepare data**, find the two
   lines of code that separate the features and
   targets, and replace them with these two lines:
   ```python
   features = chess.filter(['base_diameter','height'])
   targets = chess.filter(['piece'])
   ```

4. Below that, but above the call to `train_test_split`,
   add the following code to use a `LabelEncoder` to
   encode the character string target values as
   integers:
   ```python
   label_encoder = LabelEncoder()
   label_encoder.fit(targets.piece)
   targets = label_encoder.transform(targets.piece)
   ```

5. In section **3. Specify model**, replace the existing
   code with the following:
   ```python
   model = DecisionTreeClassifier()
   ```

6. In section **5. Evaluate model**, remove the code
   that creates the scatterplot. In a classification
   problem, the target is not a continuous numerical
   value, so a scatter plot like this is not applicable.

7. Remove section **6(a). Interpret the model**—this type 
   of classification model cannot be inspected in the
   same way a linear regression model can.

8. In section **6(b). Make predictions**, replace the code
   that assigns the variable `d` with the following:
   ```python
   d = {
     'base_diameter': [27.3, 32.7, 31, 32.1, 35.9, 37.4],
     'height': [45.7, 58.1, 65.2, 46.3, 75.6, 95.4]
   }
   ```

9. At the end of step **6(b). Make predictions**, use the
   `inverse_transform` method of the `LabelEncoder` to
   decode the integer predictions to character string
   labels:
   ```python
   label_encoder.inverse_transform(predictions)
   ```

## parsnip

1. Copy the contents of `33_parsnip_regress.R` into 
   a new file named `33_parsnip_classify.R`.

2. In section **2. Prepare data**, add the following code
   at the beginning of the section to change the `piece`
   column in the `chess` data frame to a _factor_.
   ```r
   chess <- chess %>% mutate(piece = as.factor(piece))
   ```

3. In section **3 and 4. Specify and train model**:
   - Change `linear_reg()` to
     `decision_tree(mode="classification")`
   - Change `set_engine("lm")` to `set_engine("rpart")`
   - Change the model formula to:
     `piece ~ base_diameter + height`
     
4. In section **5. Evaluate model**:
   - Replace `weight` with `piece` in two places
   - Replace `.pred` with `.pred_class` in one place

5. Remove section **6(a). Interpret the model**—this type 
   of classification model cannot be inspected in the
   same way a linear regression model can.
