import numpy as np
import sys

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('BBM411-AIN411_Assignment2_Q3_TrainingDataset.tsv', delimiter='\t', header=None)
df.columns = df.iloc[0]
df.drop(df.index[0], inplace=True)
df.head()

def apply_smoothing(matrix, alpha=0.9):
    """
    Apply Laplace smoothing to the given matrix with an adjustable alpha parameter.
    Args:
    - matrix: The matrix to be smoothed.
    - alpha: The smoothing parameter.

    Returns:
    - The smoothed matrix.
    """
    smoothed_matrix = (matrix + alpha) / (np.sum(matrix, axis=1, keepdims=True) + matrix.shape[1] * alpha)
    return smoothed_matrix


def generalized_penalty(sequence_length, letter, base_penalty=0.4):
    # Example: Exponential penalty
    return base_penalty * (2 ** sequence_length - 1) if sequence_length > 1 else 0



def get_sequence_length(i, backtrace_table, letter):
    """
    Calculate the length of consecutive states of a given letter ending at position i.

    Args:
    - i: The current position in the sequence.
    - backtrace_table: The backtrace table from the Viterbi algorithm.
    - letter: The state letter for which the sequence length is being calculated.

    Returns:
    - length: The length of the consecutive sequence of the given state.
    """
    length = 0
    for j in range(i, -1, -1):
        # Check if the state at position j is the given letter
        if backtrace_table[LettersOfSequences.index(letter)][j] == letter:
            length += 1
        else:
            break  # Break the loop if the state is not the given letter

    return length


def parse_structure(row):
    length = len(row['Sequence'])
    structure = ['U'] * length  # Initialize all as 'Unknown'

    # Helper function to update structure based on feature
    def update_structure(feature_str, symbol):
        if pd.isna(feature_str):
            return
        for feature in feature_str.split(';'):
            if feature:
                _, range_str = feature.strip().split(' ')
                start, end = map(int, range_str.split('..'))
                for i in range(start - 1, end):  # Convert to 0-based index
                    structure[i] = symbol

    # Update structure based on helix, strand, and turn
    update_structure(row['Helix'], 'H')
    update_structure(row['Beta strand'], 'E')
    update_structure(row['Turn'], 'T')

    return ''.join(structure)

# Apply the function to each row
df['structure'] = df.apply(parse_structure, axis=1)


# Extract sequences from DataFrame
aminoacid_Seq = df['Sequence'].tolist()
sequenceOFss = df['structure'].tolist()

# Define unique letters for amino acids and sequences
LettersOfAminoAcids = sorted(set(letter for aa_seq in aminoacid_Seq for letter in set(aa_seq)))
LettersOfSequences = sorted(set(letter for ss_seq in sequenceOFss for letter in set(ss_seq)))

# Initialize transition matrix, emission matrix, and state matrix
tx_matrix = np.zeros((len(LettersOfSequences), len(LettersOfSequences)))
em_matrix = np.zeros((len(LettersOfSequences), len(LettersOfAminoAcids)))
st_matrix = np.zeros((len(LettersOfSequences), 1))

# You can add more comments or documentation as needed to explain the purpose of this code section



tx_matrix = np.zeros((len(LettersOfSequences), len(LettersOfSequences)))

# Calculate transition probabilities
for sample in sequenceOFss:
    for i in range(len(sample) - 1):
        from_index = LettersOfSequences.index(sample[i])
        to_index = LettersOfSequences.index(sample[i + 1])
        tx_matrix[from_index][to_index] += 1

# Normalize transition matrix
tx_matrix /= np.sum(tx_matrix, axis=1)[:, None]

# Calculate log probabilities
tx_matrix = np.log2(tx_matrix)

# Print a message indicating the completion of training
print("Transition matrix training finished.")






# Initialize an emission matrix
em_matrix = np.zeros((len(LettersOfSequences), len(LettersOfAminoAcids)))

# Calculate emission probabilities
for sample_i in range(len(sequenceOFss)):
    ss = sequenceOFss[sample_i]
    aa = aminoacid_Seq[sample_i]
    for i in range(len(ss)):
        seq_index = LettersOfSequences.index(ss[i])
        amino_index = LettersOfAminoAcids.index(aa[i])
        em_matrix[seq_index][amino_index] += 1

# Normalize emission matrix with Laplace smoothing
em_matrix = (em_matrix + 1) / (np.sum(em_matrix, axis=1) + len(LettersOfAminoAcids))[:, None]

# Calculate log probabilities
em_matrix = np.log2(em_matrix)

# Print a message indicating the completion of training
print("Emission matrix training finished.")


# learning st_matrix

initial_letters_of_ss = [ss_seq[0] for ss_seq in sequenceOFss]
unique, counts = np.unique(np.array(initial_letters_of_ss), return_counts=True)
probs = counts / counts.sum()
for letter,prob in zip(unique, probs):
    st_matrix[LettersOfSequences.index(letter)] = prob
    
st_matrix = np.log2(st_matrix)
print("training for initial state matrix finished")

# handle command line args
measure_flag = 1


def adjust_transition_probabilities(tx_matrix, adjustments):
    """
    Modifies the transition matrix based on given adjustments and normalizes the matrix.

    Args:
    tx_matrix (numpy array): The original transition matrix.
    adjustments (dict): A dictionary where keys are tuples in the form of (from_state, to_state),
                        and values are the amounts to be added to those transitions.

    Returns:
    numpy array: The adjusted and normalized transition matrix.

    Note:
    The function assumes that the transition matrix and the adjustments are valid 
    and compatible with each other.
    """
    # Apply adjustments to the transition matrix
    for (from_state, to_state), adjustment in adjustments.items():
        from_index = LettersOfSequences.index(from_state)
        to_index = LettersOfSequences.index(to_state)
        tx_matrix[from_index][to_index] += adjustment

    # Normalize the matrix to ensure that each row sums to 1
    tx_matrix /= tx_matrix.sum(axis=1, keepdims=True)

    return tx_matrix

# Example usage
# Define adjustments and apply them to the transition matrix
adjustments = {('H', 'U'): -1, ('E', 'U'): -1, ('T', 'U'): -1, ('U', 'U'): -0.1, ('H', 'H'): -1, ('U', 'H'): -1}
# adjusted_tx_matrix = adjust_transition_probabilities(tx_matrix, adjustments)





"""
def apply_state_specific_smoothing(matrix, alpha=0.5, u_alpha=0.5):
    

    smoothed_matrix = np.copy(matrix)
    for i, state in enumerate(LettersOfSequences):
        current_alpha = u_alpha if state == 'U' else alpha
        smoothed_matrix[i, :] = (matrix[i, :] + current_alpha) / (np.sum(matrix[i, :]) + len(LettersOfAminoAcids) * current_alpha)
    
    return smoothed_matrix

# Apply this custom smoothing to your emission matrix
em_matrix = apply_state_specific_smoothing(em_matrix)


"""


print(tx_matrix)

def read_sequence_from_file(file_path):
    """
    Reads the protein information and sequence from a text file.

    Args:
    file_path (str): The path to the text file to be read.

    Returns:
    tuple: A tuple containing the protein information and the sequence.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 2:
            raise ValueError("File does not contain enough lines.")

        # Extract protein info from the first line
        protein_info = lines[0].strip()

        # Extract sequence from the second line
        sequence = lines[1].strip()

    return protein_info, sequence

def parse_protein_info(protein_info):
    """
    Parses the protein information to extract the UniProt ID and protein name.

    Args:
    - protein_info: A string containing the protein information.

    Returns:
    - tuple: A tuple containing the UniProt ID and protein name.
    """
    parts = protein_info.split('|')
    if len(parts) < 3:
        raise ValueError("Protein information is not in the expected format.")
    
    uniprot_id = parts[1]
    protein_name = parts[2].split()[0]  # Assumes protein name is the first word after the second '|'
    return uniprot_id, protein_name


def format_output(protein_info, sequence, predicted_sequence, path_prob_log):

    """
    Formats the output in the specified training dataset format.

    Args:
    - uniprot_id: UniProt ID of the protein
    - protein_name: Name of the protein
    - sequence: Amino acid sequence of the protein
    - predicted_sequence: Predicted secondary structure sequence
    - path_prob_log: Logarithm of the path probability

    Returns:
    - Formatted string
    """
    uniprot_id, protein_name = parse_protein_info(protein_info)

    def format_regions(seq, region_type):
        regions = []
        start = None
        for i, char in enumerate(seq):
            if char == region_type and start is None:
                start = i + 1
            elif char != region_type and start is not None:
                regions.append(f"{region_type} {start}..{i}")
                start = None
        if start is not None:
            regions.append(f"{region_type} {start}..{len(seq)}")
        return '; '.join(regions)

    helix_regions = format_regions(predicted_sequence, 'H')
    strand_regions = format_regions(predicted_sequence, 'E')
    turn_regions = format_regions(predicted_sequence, 'T')

    return f"{uniprot_id}\t{protein_name}\t{sequence}\t{helix_regions}\t{strand_regions}\t{turn_regions}\tProbability of path: 2^({path_prob_log})"

# Usage
file_path = 'sequence.txt'  # Replace with your actual file path
protein_info, test_seq = read_sequence_from_file(file_path)



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# VITERBI


def viterbi(state, i):
    """
    Calculates the Viterbi algorithm's probability and backtrace for a given state and position.

    Args:
    state (str): The current state.
    i (int): The current position in the test sequence.

    Returns:
    tuple: A tuple containing the probability and the letter from which the state transitioned.
    """
    emission_prob = em_matrix[LettersOfSequences.index(state)][LettersOfAminoAcids.index(test_seq[i])]
    max_term_list = []

    for j, prev_state_letter in enumerate(LettersOfSequences):
        transition_prob = tx_matrix[j][LettersOfSequences.index(state)]

        # Apply penalty for repeated states
        if i > 0 and backtrace_table[j][i - 1] == state:
            sequence_length = get_sequence_length(i - 1, backtrace_table, state)
            transition_prob -= generalized_penalty(sequence_length, state)

        max_term_list.append(partial_scores_table[j][i - 1] + transition_prob)

    max_probability = emission_prob + max(max_term_list)
    backtrace_from_index = np.argmax(np.array(max_term_list))

    return max_probability, LettersOfSequences[backtrace_from_index]

# Viterbi Initialization
partial_scores_table = np.zeros((len(LettersOfSequences), len(test_seq)))
backtrace_table = np.zeros((len(LettersOfSequences), len(test_seq)), dtype=object)

for state in LettersOfSequences:
    state_index = LettersOfSequences.index(state)
    aa_index = LettersOfAminoAcids.index(test_seq[0])
    initial_state_prob = st_matrix[state_index][0] 
    initial_emission_prob = em_matrix[state_index][aa_index]
    partial_scores_table[state_index][0] = initial_state_prob + initial_emission_prob

# Fill Partial Score Table
for i in range(1, len(test_seq)):
    for state in LettersOfSequences:
        prob, backtrace_from = viterbi(state, i)
        state_index = LettersOfSequences.index(state)
        partial_scores_table[state_index][i] = prob
        backtrace_table[state_index][i] = backtrace_from







def find_last_column_max_index(partial_scores_table):
    """
    Identifies the index of the maximum value in the last column of a partial scores table.

    Args:
    partial_scores_table (numpy array): A 2D numpy array representing the partial scores table.

    Returns:
    int: The index of the maximum value in the last column of the partial scores table.
    """
    return np.argmax(partial_scores_table[:, -1])

def backtrack_for_prediction(backtrace_table, letters_of_sequences, start_index):
    """
    Constructs the predicted sequence using backtracking from a given start index in a backtrace table.

    Args:
    backtrace_table (numpy array): A 2D numpy array representing the backtrace table.
    letters_of_sequences (list): A list of characters representing the sequence.
    start_index (int): The starting index for the backtracking process.

    Returns:
    str: The predicted sequence obtained by backtracking through the backtrace table.
    """
    predicted_sequence = [letters_of_sequences[start_index]]
    for i in range(backtrace_table.shape[1] - 1, 0, -1):
        current_letter = backtrace_table[start_index, i]
        predicted_sequence.append(current_letter)
        start_index = letters_of_sequences.index(current_letter)
    
    return ''.join(reversed(predicted_sequence))

# Example usage
last_col_max_index = find_last_column_max_index(partial_scores_table)
path_prob_log = partial_scores_table[last_col_max_index, -1]
pred_sequenceOFss = backtrack_for_prediction(backtrace_table, LettersOfSequences, last_col_max_index)





# write to output file






#--------------------------------------------------------------------------------
# MEASURE PERFORMANCE

# read ground truth secondary structure sequence of corresponding protein
def read_file(file_path):
    """
    Reads a text file and returns its lines with trailing spaces removed.

    Args:
    file_path (str): The path to the text file to be read.

    Returns:
    list: A list of strings, where each string is a line from the file, stripped of trailing spaces.
    """
    with open(file_path, 'r') as file:
        lines = [line.rstrip() for line in file]

    return lines

def parse_regions(lines):
    """
    Parses a list of strings into a structured list of regions. Each region is represented as a list with the region type
    and a list of start and end indices. Adjusts 'S' region types to 'E'.

    Args:
    lines (list of str): The lines of text representing the regions.

    Returns:
    list: A list of regions, each represented as [region_type, [start, end]].
    """
    regions = []
    for line in lines:
        elements = line.split()
        region_type = 'E' if elements[0][0] == 'S' else elements[0][0]
        start, end = int(elements[1]) - 1, int(elements[2]) - 1
        regions.append([region_type, [start, end]])

    return regions

def generate_sequence(test_seq_length, regions):
    """
    Generates a sequence of a specified length, initializing with '_', and updates it based on provided regions.

    Args:
    test_seq_length (int): The length of the sequence to generate.
    regions (list): A list of regions, where each region is represented as [region_type, [start, end]].

    Returns:
    list: A sequence list where each element is either '_' or one of the specified region types.
    """
    sequence = ['_'] * test_seq_length
    for region_type, (start, end) in regions:
        sequence[start:end + 1] = [region_type] * (end - start + 1)

    return sequence

# Main execution block
if measure_flag == 1:
    lines = read_file("gt-tp53.txt")
    gt_regions = parse_regions(lines)
    gt_sequenceOFss = generate_sequence(len(test_seq), gt_regions)

 

    # performance calculations
def calculate_confusion_matrix_row(gt_element, gt_sequence, pred_sequence):
    """
    Calculates the counts of predictions for a given ground truth element in the confusion matrix row.

    Args:
    gt_element (str): The ground truth element to compare against (H, E, T, or U).
    gt_sequence (list/str): The sequence of ground truth labels.
    pred_sequence (list/str): The sequence of predicted labels.

    Returns:
    tuple: A tuple containing the counts of predictions as H, E, T, U for the given ground truth element.
    """
    # Replace '_' with 'U' in ground truth sequence
    gt_sequence = ['U' if label == '_' else label for label in gt_sequence]

    # Calculate counts for each prediction category
    count_pred_as_H = sum(gt == gt_element and pred == 'H' for gt, pred in zip(gt_sequence, pred_sequence))
    count_pred_as_E = sum(gt == gt_element and pred == 'E' for gt, pred in zip(gt_sequence, pred_sequence))
    count_pred_as_T = sum(gt == gt_element and pred == 'T' for gt, pred in zip(gt_sequence, pred_sequence))
    count_pred_as_U = sum(gt == gt_element and pred == 'U' for gt, pred in zip(gt_sequence, pred_sequence))

    return count_pred_as_H, count_pred_as_E, count_pred_as_T, count_pred_as_U


def safe_divide(numerator, denominator):
	return numerator / denominator if denominator != 0 else 0


hh, he, ht, hu = calculate_confusion_matrix_row('H', gt_sequenceOFss, pred_sequenceOFss)
eh, ee, et, eu = calculate_confusion_matrix_row('E', gt_sequenceOFss, pred_sequenceOFss)
th, te, tt, tu = calculate_confusion_matrix_row('T', gt_sequenceOFss, pred_sequenceOFss)
uh, ue, ut, uu = calculate_confusion_matrix_row('U', gt_sequenceOFss, pred_sequenceOFss)

# True Positives
tp_h = hh
tp_e = ee
tp_t = tt
tp_u = uu

# True Negatives
tn_h = ee + et + eu + te + tt + tu + ue + ut + uu
tn_e = hh + ht + hu + th + tt + tu + uh + ut + uu
tn_t = hh + he + hu + eh + ee + eu + uh + ue + uu
tn_u = hh + he + ht + eh + ee + et + th + te + tt

# False Positives
fp_h = eh + th + uh
fp_e = he + te + ue
fp_t = ht + et + ut
fp_u = hu + eu + tu

# False Negatives
fn_h = he + ht + hu
fn_e = eh + et + eu
fn_t = th + te + tu
fn_u = uh + ue + ut

# Precision, Recall, and F1 Score
prec_h = safe_divide(tp_h, tp_h + fp_h)
prec_e = safe_divide(tp_e, tp_e + fp_e)
prec_t = safe_divide(tp_t, tp_t + fp_t)
prec_u = safe_divide(tp_u, tp_u + fp_u)

recall_h = safe_divide(tp_h, tp_h + fn_h)
recall_e = safe_divide(tp_e, tp_e + fn_e)
recall_t = safe_divide(tp_t, tp_t + fn_t)
recall_u = safe_divide(tp_u, tp_u + fn_u)

f1_h = safe_divide(2 * prec_h * recall_h, prec_h + recall_h)
f1_e = safe_divide(2 * prec_e * recall_e, prec_e + recall_e)
f1_t = safe_divide(2 * prec_t * recall_t, prec_t + recall_t)
f1_u = safe_divide(2 * prec_u * recall_u, prec_u + recall_u)

# Accuracy
acc_h = safe_divide(tp_h + tn_h, tp_h + tn_h + fp_h + fn_h)
acc_e = safe_divide(tp_e + tn_e, tp_e + tn_e + fp_e + fn_e)
acc_t = safe_divide(tp_t + tn_t, tp_t + tn_t + fp_t + fn_t)
acc_u = safe_divide(tp_u + tn_u, tp_u + tn_u + fp_u + fn_u)

# Print Confusion Matrix and Metrics
print('\t\t\tPredicted')
print('\t\t\tH\tE\tT\tU')
print('\t\tH\t' + str(hh) + '\t' + str(he) + '\t' + str(ht) + '\t' + str(hu))
print('Ground Truth\tE\t' + str(eh) + '\t' + str(ee) + '\t' + str(et) + '\t' + str(eu))
print('\t\tT\t' + str(th) + '\t' + str(te) + '\t' + str(tt) + '\t' + str(tu))
print('\t\tU\t' + str(uh) + '\t' + str(ue) + '\t' + str(ut) + '\t' + str(uu))

print("\n")
print("\tPrec\t\tRecall\t\tF1\t\tAcc")
print("H\t" + str(round(prec_h, 4)) + "\t\t" + str(round(recall_h, 4)) + "\t\t" + str(round(f1_h, 4)) + "\t\t" + str(round(acc_h, 4)))
print("E\t" + str(round(prec_e, 4)) + "\t\t" + str(round(recall_e, 4)) + "\t\t" + str(round(f1_e, 4)) + "\t\t" + str(round(acc_e, 4)))
print("T\t" + str(round(prec_t, 4)) + "\t\t" + str(round(recall_t, 4)) + "\t\t" + str(round(f1_t, 4)) + "\t\t" + str(round(acc_t, 4)))
print("U\t" + str(round(prec_u, 4)) + "\t\t" + str(round(recall_u, 4)) + "\t\t" + str(round(f1_u, 4)) + "\t\t" + str(round(acc_u, 4)))

# Overall Accuracy
total_correct = hh + ee + tt + uu
total_predictions = sum([hh, he, ht, hu, eh, ee, et, eu, th, te, tt, tu, uh, ue, ut, uu])
print('\nOverall Accuracy: ' + str(round(total_correct / total_predictions, 4)))



predicted_sequence = pred_sequenceOFss  # This should be the predicted sequence from your Viterbi algorithm
path_prob_log = path_prob_log  # The logarithm of the path probability from your Viterbi algorithm

formatted_output = format_output(protein_info, test_seq, predicted_sequence, path_prob_log)
print(formatted_output)

with open('output2.txt', 'w') as file:
    file.write(formatted_output)

print("Formatted output saved to output2.txt")

def calculate_confusion_matrix_for_u(pred_seq, gt_seq):
    """
    Calculates the confusion matrix values specifically for the 'U' category in the predictions.

    Args:
    pred_seq (list/str): The sequence of predicted labels.
    gt_seq (list/str): The ground truth sequence of labels.

    Returns:
    tuple: A tuple containing the counts of true positives, false negatives, and false positives for the 'U' category.
    """
    hu, eu, tu, uu = 0, 0, 0, 0
    for gt_label, pred_label in zip(gt_seq, pred_seq):
        if gt_label != 'U' and pred_label == 'U':
            if gt_label == 'H':
                hu += 1
            elif gt_label == 'E':
                eu += 1
            elif gt_label == 'T':
                tu += 1
        elif gt_label == 'U' and pred_label == 'U':
            uu += 1

    return hu, eu, tu, uu

# Example usage
hu, eu, tu, uu = calculate_confusion_matrix_for_u(pred_sequenceOFss, gt_sequenceOFss)



# confusion matrix
'''
df_cm = pd.DataFrame([[hh,he,ht, hu], [eh, ee, et, eu], [th, te, tt, tu], [uh, ue, ut, uu]], index = ['H', 'E', 'T', 'U'], columns = ['H', 'E', 'T', 'U'])
ax = sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix\n", fontsize =15)
plt.xlabel('Predicted', fontsize = 12)
plt.ylabel('Ground Truth', fontsize = 12)
plt.show()
''' 