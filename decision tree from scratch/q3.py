import numpy as np
import pandas as pd
from collections import Counter

# Global variables
full_tree = None
category_mappings = {}
feature_names = ['MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK', 'DISTANCE_GROUP', 'NUMBER_OF_SEATS',
                 'CONCURRENT_FLIGHTS', 'AIRPORT_FLIGHTS_MONTH', 'AIRLINE_FLIGHTS_MONTH',
                 'SEASON', 'DAY_TYPE', 'TIME_BLOCK', 'DISTANCE_CATEGORY', 'SEAT_CATEGORY',
                 'CONCURRENT_CATEGORY', 'AIRPORT_TRAFFIC', 'AIRLINE_TRAFFIC']#
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, branches=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.branches = branches if branches is not None else {}

# Utility functions for categorization
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def get_day_type(day):
    return 'Weekend' if day in [6, 7] else 'Weekday'

def get_time_block(time_blk):
    if time_blk in ['0600-0659', '0700-0759', '0800-0859', '0900-0959', '1000-1059', '1100-1159']:
        return 'Morning'
    elif time_blk in ['1200-1259', '1300-1359', '1400-1459', '1500-1559', '1600-1659', '1700-1759']:
        return 'Afternoon'
    elif time_blk in ['1800-1859', '1900-1959', '2000-2059', '2100-2159', '2200-2259', '2300-2359']:
        return 'Evening'
    else:
        return 'Night'

def get_distance_category(group):
    if group in [1, 2, 3]:
        return 'Short-haul'
    elif group in [4, 5, 6]:
        return 'Medium-haul'
    else:
        return 'Long-haul'

def get_seat_category(seats):
    if seats < 100:
        return 'Small'
    elif 100 <= seats <= 200:
        return 'Medium'
    else:
        return 'Large'

def get_concurrent_category(flights):
    if flights < 10:
        return 'Low Traffic'
    elif 10 <= flights <= 20:
        return 'Medium Traffic'
    else:
        return 'High Traffic'

def get_airport_traffic_category(flights):
    if flights < 10000:
        return 'Low Traffic Airport'
    elif 10000 <= flights <= 50000:
        return 'Medium Traffic Airport'
    else:
        return 'High Traffic Airport'

def get_airline_traffic_category(flights):
    if flights < 5000:
        return 'Low Traffic Airline'
    elif 5000 <= flights <= 25000:
        return 'Medium Traffic Airline'
    else:
        return 'High Traffic Airline'

# Entropy and Information Gain functions
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def information_gain(X, y, feature_index, num_thresholds=10):
    parent_entropy = entropy(y)
    values = X[:, feature_index]
    thresholds = np.percentile(values, np.linspace(0, 100, num_thresholds + 2)[1:-1])
    best_gain = 0
    best_threshold = None

    for threshold in thresholds:
        left_indices = values <= threshold
        right_indices = values > threshold
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            continue

        left_entropy = entropy(y[left_indices])
        right_entropy = entropy(y[right_indices])
        weighted_avg_entropy = (len(y[left_indices]) * left_entropy + len(y[right_indices]) * right_entropy) / len(y)
        gain = parent_entropy - weighted_avg_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold

def information_gain_categorical(X, y, feature_index):
    parent_entropy = entropy(y)
    categories = np.unique(X[:, feature_index])
    weighted_avg_entropy = 0

    for category in categories:
        subset = y[X[:, feature_index] == category]
        weighted_avg_entropy += (len(subset) / len(y)) * entropy(subset)

    gain = parent_entropy - weighted_avg_entropy
    return gain

# Tree building and prediction functions
def build_tree_f(X, y, types, used_features=set(), depth=0, max_depth=9, branches_num=0):
    num_samples, num_features = X.shape
    if num_samples <= 20 or depth >= max_depth or len(np.unique(y)) == 1 or branches_num >= 20:
        leaf_value = np.bincount(y).argmax()
        return DecisionTreeNode(value=leaf_value)

    best_gain = 0
    best_feature_index = None
    best_threshold = None

    for feature_index in range(num_features):
        if feature_index in used_features:
            continue
        if types[feature_index] == 'continuous':
            gain, threshold = information_gain(X, y, feature_index)
        elif len(np.unique(X[:, feature_index])) < 8:  # Categorical feature
            gain = information_gain_categorical(X, y, feature_index)
            threshold = None

        if gain > best_gain:
            best_gain = gain
            best_feature_index = feature_index
            best_threshold = threshold

    if best_gain == 0:
        leaf_value = np.bincount(y).argmax()
        return DecisionTreeNode(value=leaf_value)

    used_features.add(best_feature_index)

    node = DecisionTreeNode(feature_index=best_feature_index, threshold=best_threshold)
    node.value = np.bincount(y).argmax()  # Store the most common class in this node

    if best_threshold is not None:  # Numerical feature
        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = X[:, best_feature_index] > best_threshold

        node.left = build_tree_f(X[left_indices], y[left_indices], types, used_features.copy(), depth + 1, max_depth)
        node.right = build_tree_f(X[right_indices], y[right_indices], types, used_features.copy(), depth + 1, max_depth)
    else:  # Categorical feature
        node.branches = {}
        categories = np.unique(X[:, best_feature_index])
        branches_number = len(categories)

        for category in categories:
            indices = X[:, best_feature_index] == category
            node.branches[category] = build_tree_f(X[indices], y[indices], types, used_features.copy(), depth + 1,
                                                   max_depth, branches_num=branches_number)

    return node

def predict(tree, x):
    if tree.left is None and tree.right is None and not tree.branches:
        return tree.value

    feature_value = x[tree.feature_index]
    if tree.threshold is not None:  # Numerical feature
        if feature_value <= tree.threshold:
            return predict(tree.left, x)
        else:
            return predict(tree.right, x)
    elif tree.branches is not None:  # Categorical feature
        branch = tree.branches.get(feature_value, None)
        if branch is not None:
            return predict(branch, x)

    # If we reach here, it means we've encountered an unseen category
    # or some other unexpected situation. Return the most common class for this node.
    return tree.value

# Data preprocessing functions
def divide_continous_factorial(ndarray):
    types = []
    for i in range(ndarray.shape[1]):
        unique_values = np.unique(ndarray[:, i])
        if all(isinstance(value, str) for value in unique_values):
            types.append('categorical')
        elif len(unique_values) > 7:
            types.append('continuous')
        else:
            types.append('categorical')
    return types

def handle_missing_values(X, types):
    for i in range(X.shape[1]):
        if types[i] == 'continuous':
            # For numerical features, use mean imputation
            mean_value = np.nanmean(X[:, i].astype(float))
            X[:, i] = np.where(pd.isnull(X[:, i]), mean_value, X[:, i])
        else:
            # For categorical features, treat missing values as a separate category
            X[:, i] = np.where(pd.isnull(X[:, i]), 'MISSING', X[:, i])
    return X

def group_categories_min_entropy(X, y, column_index, max_categories=8):
    global category_mappings
    original_categories = np.unique(X[:, column_index])
    categories = original_categories.copy()

    if len(categories) <= max_categories or len(categories) > 20:
        return X

    # Calculate initial entropy for each category
    category_entropy = {}
    for cat in categories:
        category_entropy[cat] = entropy(y[X[:, column_index] == cat])

    # Create a dictionary to map categories to their counts
    category_counts = {cat: np.sum(X[:, column_index] == cat) for cat in categories}

    while len(categories) > max_categories:
        # Find the pair of categories that when merged have the minimum increase in entropy
        min_increase = float('inf')
        merge_pair = None

        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                merged_entropy = entropy(np.concatenate([y[X[:, column_index] == cat1], y[X[:, column_index] == cat2]]))
                increase = merged_entropy - (category_entropy[cat1] + category_entropy[cat2])

                if increase < min_increase:
                    min_increase = increase
                    merge_pair = (cat1, cat2)

        # Merge the pair with minimum increase in entropy
        cat1, cat2 = merge_pair
        new_cat = f"{cat1}, {cat2}"
        X[:, column_index][(X[:, column_index] == cat1) | (X[:, column_index] == cat2)] = new_cat

        # Update categories and entropies
        categories = np.array([cat for cat in categories if cat not in merge_pair] + [new_cat])
        category_entropy[new_cat] = entropy(y[X[:, column_index] == new_cat])
        del category_entropy[cat1]
        del category_entropy[cat2]

    # Store the mapping
    category_mappings[column_index] = {}
    for orig_cat in original_categories:
        for new_cat in categories:
            if orig_cat in new_cat.split(', '):
                category_mappings[column_index][orig_cat] = new_cat
                break
        else:
            category_mappings[column_index][orig_cat] = 'MISSING'

    return X

def encode_high_cardinality(X, column_index, top_n=20):
    column = X[:, column_index]
    value_counts = Counter(column)
    top_categories = dict(value_counts.most_common(top_n))

    def encode(val):
        if val in top_categories:
            return str(val)
        else:
            return 'Other'

    X[:, column_index] = np.array([encode(val) for val in column])
    return X

def rank_by_popularity(series):
    """Rank categories by popularity (frequency), with 1 being the most popular"""
    value_counts = series.value_counts()
    ranks = pd.Series(range(1, len(value_counts) + 1), index=value_counts.index)
    return series.map(ranks)
def preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure all required columns exist
    required_columns = ['MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK', 'DISTANCE_GROUP', 'NUMBER_OF_SEATS',
                        'CONCURRENT_FLIGHTS', 'AIRPORT_FLIGHTS_MONTH', 'AIRLINE_FLIGHTS_MONTH',
                        'CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file.")

    # Apply categorization functions and replace values in original columns
    df['MONTH'] = df['MONTH'].apply(get_season)
    df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].apply(get_day_type)
    df['DEP_TIME_BLK'] = df['DEP_TIME_BLK'].apply(get_time_block)
    df['DISTANCE_GROUP'] = df['DISTANCE_GROUP'].apply(get_distance_category)
    df['NUMBER_OF_SEATS'] = df['NUMBER_OF_SEATS'].apply(get_seat_category)
    df['CONCURRENT_FLIGHTS'] = df['CONCURRENT_FLIGHTS'].apply(get_concurrent_category)
    df['AIRPORT_FLIGHTS_MONTH'] = df['AIRPORT_FLIGHTS_MONTH'].apply(get_airport_traffic_category)
    df['AIRLINE_FLIGHTS_MONTH'] = df['AIRLINE_FLIGHTS_MONTH'].apply(get_airline_traffic_category)

    # Rank airlines and airports by popularity
    df['CARRIER_NAME'] = rank_by_popularity(df['CARRIER_NAME'])
    df['DEPARTING_AIRPORT'] = rank_by_popularity(df['DEPARTING_AIRPORT'])
    df['PREVIOUS_AIRPORT'] = rank_by_popularity(df['PREVIOUS_AIRPORT'])

    # Convert dataframe to ndarray
    ndarray = df.values

    # Determine types (numerical or categorical) for each feature
    types = divide_continous_factorial(ndarray)

    # Ensure CARRIER_NAME, DEPARTING_AIRPORT, and PREVIOUS_AIRPORT are treated as continuous
    carrier_name_index = df.columns.get_loc('CARRIER_NAME')
    departing_airport_index = df.columns.get_loc('DEPARTING_AIRPORT')
    previous_airport_index = df.columns.get_loc('PREVIOUS_AIRPORT')
    types[carrier_name_index] = 'continuous'
    types[departing_airport_index] = 'continuous'
    types[previous_airport_index] = 'continuous'

    # Handle missing values
    ndarray = handle_missing_values(ndarray, types)

    return ndarray, types


#Tree pruning functions
def chi_squared_test(node, X, y):
    if not node.branches and (node.left is None or node.right is None):
        return False  # No pruning if there are no children

    subsets = []
    if node.threshold is not None:  # Numerical feature
        left_indices = X[:, node.feature_index] <= node.threshold
        right_indices = X[:, node.feature_index] > node.threshold
        subsets = [left_indices, right_indices]
    else:  # Categorical feature
        subsets = [X[:, node.feature_index] == category for category in node.branches]

    total_counts = Counter(y)
    total_size = len(y)
    chi_squared_stat = 0

    for subset in subsets:
        subset_counts = Counter(y[subset])
        subset_size = len(y[subset])
        expected_counts = {cls: (subset_size / total_size) * count for cls, count in total_counts.items()}

        for cls in total_counts.keys():
            observed = subset_counts.get(cls, 0)
            expected = expected_counts.get(cls, 0)
            if expected > 0:
                chi_squared_stat += (observed - expected) ** 2 / expected

    df = len(total_counts) - 1
    critical_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}
    critical_value = critical_values.get(df, 9.488)

    return chi_squared_stat < critical_value


def prune_tree(node, X, y):
    if node.left is not None:
        left_indices = X[:, node.feature_index] <= node.threshold
        prune_tree(node.left, X[left_indices], y[left_indices])
    if node.right is not None:
        right_indices = X[:, node.feature_index] > node.threshold
        prune_tree(node.right, X[right_indices], y[right_indices])

    if node.branches:
        for category, branch in node.branches.items():
            indices = X[:, node.feature_index] == category
            prune_tree(branch, X[indices], y[indices])

    if chi_squared_test(node, X, y):
        node.left = None
        node.right = None
        node.branches = {}
        node.value = np.bincount(y).argmax()




def preprocess_prediction_data(x, category_mappings):
    x = list(x)  # Convert to list for easier manipulation

    # Apply categorization functions
    x[0] = get_season(x[0])  # MONTH
    x[1] = get_day_type(x[1])  # DAY_OF_WEEK
    x[2] = get_time_block(x[2])  # DEP_TIME_BLK
    x[3] = get_distance_category(x[3])  # DISTANCE_GROUP
    x[6] = get_seat_category(x[6])  # NUMBER_OF_SEATS
    x[5] = get_concurrent_category(x[5])  # CONCURRENT_FLIGHTS
    x[8] = get_airport_traffic_category(x[8])  # AIRPORT_FLIGHTS_MONTH
    x[9] = get_airline_traffic_category(x[9])  # AIRLINE_FLIGHTS_MONTH

    # Handle CARRIER_NAME, DEPARTING_AIRPORT, and PREVIOUS_AIRPORT
    carrier_name_index = 7
    departing_airport_index = 16
    previous_airport_index = 19

    # If these mappings exist in category_mappings, use them. Otherwise, assign a high rank (less popular)
    x[carrier_name_index] = category_mappings.get('CARRIER_NAME', {}).get(x[carrier_name_index], len(category_mappings.get('CARRIER_NAME', {})) + 1)
    x[departing_airport_index] = category_mappings.get('DEPARTING_AIRPORT', {}).get(x[departing_airport_index], len(category_mappings.get('DEPARTING_AIRPORT', {})) + 1)
    x[previous_airport_index] = category_mappings.get('PREVIOUS_AIRPORT', {}).get(x[previous_airport_index], len(category_mappings.get('PREVIOUS_AIRPORT', {})) + 1)

    # Apply other category mappings
    for col_name, mapping in category_mappings.items():
        if col_name not in ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']:
            col_index = feature_names.index(col_name)
            if col_index < len(x):
                x[col_index] = mapping.get(x[col_index], 'NONE')

    return x

# Update the is_late function to create category_mappings correctly
def is_late(row_input):
    global full_tree, category_mappings

    if full_tree is None:
        file_path = 'flightdelay.csv'
        df = pd.read_csv(file_path)
        ndarray, types = preprocess_data(file_path)
        X = ndarray[:, :-1]
        y = ndarray[:, -1].astype(int)

        # Build the decision tree on the full dataset
        full_tree = build_tree_f(X, y, types)
        prune_tree(full_tree, X, y)

        # Create category_mappings for all categorical columns
        category_mappings = {}
        for col in df.columns:
            if col in ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']:
                category_mappings[col] = dict(zip(df[col].unique(), rank_by_popularity(df[col])))
            elif df[col].dtype == 'object':  # For other categorical columns
                category_mappings[col] = dict(zip(df[col].unique(), range(len(df[col].unique()))))

    row_input = np.array(row_input)
    row_input = preprocess_prediction_data(row_input, category_mappings)

    # Predict if the given row input will result in a late flight
    prediction = predict(full_tree, row_input)
    return prediction


def print_tree(node, feature_names=None, indent="", feature_name=""):
    if node is None:
        return

    if node.left is None and node.right is None and not node.branches:
        print(f"{indent}{feature_name}: Leaf Node with value {node.value}")
        return

    if feature_names is not None:
        feature_name = feature_names[node.feature_index]
    else:
        feature_name = f"Feature {node.feature_index}"

    if node.threshold is not None:
        print(f"{indent}{feature_name} <= {node.threshold:.2f}")
        print_tree(node.left, feature_names, indent + "  │   ", "Yes")
        print(f"{indent}  └── {feature_name} > {node.threshold:.2f}")
        print_tree(node.right, feature_names, indent + "      ", "No")
    else:
        print(f"{indent}{feature_name}")
        for category, branch in node.branches.items():
            print(f"{indent}  ├── {category}")
            print_tree(branch, feature_names, indent + "  │   ")

        # Print the last branch with a different character to close the tree properly
        last_category = list(node.branches.keys())[-1]
        print(f"{indent}  └── {last_category}")
        print_tree(node.branches[last_category], feature_names, indent + "      ")


def k_fold_split(X, y, k):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = len(X) // k
    folds = []
    for i in range(k):
        start = i * fold_sizes
        end = (i + 1) * fold_sizes if i != k - 1 else len(X)
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((X[train_indices], X[test_indices], y[train_indices], y[test_indices]))
    return folds


def tree_error(k):
    file_path = 'flightdelay.csv'
    ndarray, types = preprocess_data(file_path)

    X = ndarray[:, :-1]
    y = ndarray[:, -1].astype(int)

    folds = k_fold_split(X, y, k)
    errors = []
    num = 1
    for train_X, test_X, train_y, test_y in folds:
        tree = build_tree_f(train_X, train_y, types)
        print(f"Decision Tree Structure number {num} before pruning \n \n \n")
        #print_tree(tree, feature_names)
        prune_tree(tree, train_X, train_y)
        predictions = [predict(tree, x) for x in test_X]
        print(f"Decision Tree Structure number {num}")
        #print_tree(tree, feature_names)
        num = num + 1
        error = np.mean(predictions != test_y)
        errors.append(error)
    average_error = np.mean(errors)
    print(f"Cross-validation Error: {average_error}")
    return average_error


def build_tree(ratio):
    file_path = 'flightdelay.csv'
    ndarray, types = preprocess_data(file_path)

    X = ndarray[:, :-1]
    y = ndarray[:, -1].astype(int)

    # Split the data into training and validation sets based on the ratio
    num_train = int(len(X) * ratio)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Build the decision tree
    tree = build_tree_f(X_train, y_train, types)

    # Prune the decision tree
    prune_tree(tree, X_train, y_train)
    print_tree(tree, feature_names)

    # Validate the decision tree
    predictions = [predict(tree, x) for x in X_val]
    error = np.mean(predictions != y_val)

    # Report the error
    print(f"\nValidation Error: {error}")

    return tree, error


# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('flightdelay.csv')

    # Feature names (adjust these based on your actual column names)
    feature_names = ['MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK', 'DISTANCE_GROUP', 'SEGMENT_NUMBER',
                     'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS', 'CARRIER_NAME', 'AIRPORT_FLIGHTS_MONTH',
                     'AIRLINE_FLIGHTS_MONTH', 'AIRLINE_AIRPORT_FLIGHTS_MONTH', 'AVG_MONTHLY_PASS_AIRPORT',
                     'AVG_MONTHLY_PASS_AIRLINE', 'FLT_ATTENDANTS_PER_PASS', 'GROUND_SERV_PER_PASS',
                     'PLANE_AGE', 'DEPARTING_AIRPORT', 'LATITUDE', 'LONGITUDE', 'PREVIOUS_AIRPORT',
                     'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND']

    # Perform k-fold cross-validation
    print("Performing 5-fold cross-validation:")
   # tree_error(5)

    # Build a tree with 60% training data
    print("\nBuilding a tree with 60% training data:")
    #build_tree(0.6)

    # Make predictions on individual rows
    print("\nMaking predictions on individual rows:")
    print(f"Is flight in row 3 late? {is_late(df.iloc[3])}")
    print(f"Is flight in row 10 late? {is_late(df.iloc[10])}")
    print(f"Is flight in row 20 late? {is_late(df.iloc[20])}")