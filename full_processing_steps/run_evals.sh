#!/bin/bash
# This script runs the evaluation scripts for the different models, relying on openai evals library
# MODIFY THIS SCRIPT ACCORDINGLY WITH YOUR PATHS AND SIMILAR.

# IF USING A VIRTUAL ENVIRONMENT, ACTIVATE IT HERE
source ~/venv/<your_venv>/bin/activate

# SPECIFY THE PATH OF THE FOLDER WHERE THIS SCRIPT IS LOCATED
ROOT_FOLDER_PATH=<your_path>

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
# create general log file:
mkdir -p ${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}
GENERAL_OUT=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/general.out
touch $GENERAL_OUT

# load the API key
# OPEN THE .env FILE AND ADD THE API KEY!
source ${ROOT_FOLDER_PATH}/.env  

#MODELS=( "gpt-4o-2024-08-06" "gpt-4o-2024-05-13" "gpt-4o-mini-2024-07-18" "gpt-4-turbo-2024-04-09" "gpt-4-0125-preview" "gpt-4-1106-preview" "gpt-4-0613" "gpt-4-0314" "gpt-3.5-turbo-0125" "gpt-3.5-turbo-1106" "gpt-3.5-turbo-0613" "gpt-3.5-turbo-0301" )
MODELS=( "gpt-3.5-turbo-0125" )

# SET THIS TO 1 TO SKIP EVALUATIONS IF THE FILE ALREADY EXISTS; SET TO 0 TO RUN ALL EVALUATIONS AND OVERWRITE EXISTING FILES
SKIP_IF_FILE_FOUND=1  

# --- cheap evals ---
# those for which we run all samples
EVALS_CHEAP=( "odd_one_out" "cause_and_effect_one_sentence" "cause_and_effect_two_sentences" "crass_ai" "logical_args" "emoji_movie" "fantasy_reasoning" "metaphor_boolean" "geometric_shapes" "space_nli" "abstract_narrative_understanding_4_distractors" "arithmetic_1_digit_division" "arithmetic_1_digit_subtraction" "arithmetic_1_digit_addition" "arithmetic_1_digit_multiplication" "arithmetic_2_digit_division" "arithmetic_3_digit_division" "arithmetic_2_digit_multiplication" "arithmetic_2_digit_addition" "arithmetic_2_digit_subtraction" "arithmetic_3_digit_multiplication" "arithmetic_3_digit_addition" "arithmetic_3_digit_subtraction" "arithmetic_4_digit_multiplication" "arithmetic_4_digit_addition" "arithmetic_4_digit_subtraction" "arithmetic_4_digit_division" "arithmetic_5_digit_multiplication" "arithmetic_5_digit_addition" "arithmetic_5_digit_subtraction" "arithmetic_5_digit_division" "copa" "anli" "cosmos_qa" "ropes" )
MAX_SAMPLES=10000

# create empty log file:
CHEAP_EVALS_OUT=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/cheap_evals.out
CHEAP_EVALS_ERR=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/cheap_evals.err
touch $CHEAP_EVALS_OUT
touch $CHEAP_EVALS_ERR

i=0
for eval in "${EVALS_CHEAP[@]}"; do
  i=$((i+1))
  for model in "${MODELS[@]}"; do

    if [ $SKIP_IF_FILE_FOUND -eq 1 ]; then
      if [ -f ${ROOT_FOLDER_PATH}/2_results/$eval/$model.jsonl ]; then
        echo "Skipping eval $eval for model $model because file already exists" >> $GENERAL_OUT
        continue
      fi
    fi
    echo "Running eval $eval for model $model" >> $GENERAL_OUT
    oaieval $model $eval \
      --max_samples $MAX_SAMPLES \
      --record_path=${ROOT_FOLDER_PATH}/2_results/$eval/$model.jsonl \
      --registry_path ${ROOT_FOLDER_PATH}/1_registry >> $CHEAP_EVALS_OUT 2>> $CHEAP_EVALS_ERR
  done
done
echo Run $i evals >> $GENERAL_OUT

# --- expensive evals ---
EVALS_MAX_1000=("goal_step_wikihow_goal_inference" "alpha_nli" "goal_step_wikihow_step_inference" "abstract_narrative_understanding_9_distractors" "goal_step_wikihow_step_ordering" "wanli" "babi_task_16" "abstract_narrative_understanding_99_distractors" "formal_fallacies_syllogisms_negation")
MAX_SAMPLES=1000

# create empty log file:
EXPENSIVE_EVALS_OUT=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/expensive_evals.out
EXPENSIVE_EVALS_ERR=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/expensive_evals.err
touch $EXPENSIVE_EVALS_OUT
touch $EXPENSIVE_EVALS_ERR

i=0
for eval in "${EVALS_MAX_1000[@]}"; do
  i=$((i+1))
  for model in "${MODELS[@]}"; do

    if [ $SKIP_IF_FILE_FOUND -eq 1 ]; then
      if [ -f ${ROOT_FOLDER_PATH}/2_results/$eval/$model.jsonl ]; then
        echo "Skipping eval $eval for model $model because file already exists" >> $GENERAL_OUT
        continue
      fi
    fi

    echo "Running eval $eval for model $model" >> $GENERAL_OUT
    oaieval $model $eval \
      --max_samples $MAX_SAMPLES \
      --record_path=${ROOT_FOLDER_PATH}/2_results/$eval/$model.jsonl \
      --registry_path ${ROOT_FOLDER_PATH}/1_registry >> $EXPENSIVE_EVALS_OUT 2>> $EXPENSIVE_EVALS_ERR
   done
done
echo Run $i evals  >> $GENERAL_OUT

# --- evals already in library ---
EVALS_IN_LIBRARY=('reclor-logical-reasoning-plus' 'logiqav2-logical-reasoning-plus')
MAX_SAMPLES=10000

# create empty log file:
IN_LIBRARY_EVALS_OUT=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/in_library_evals.out
IN_LIBRARY_EVALS_ERR=${ROOT_FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/in_library_evals.err
touch $IN_LIBRARY_EVALS_OUT
touch $IN_LIBRARY_EVALS_ERR

i=0
for eval in "${EVALS_IN_LIBRARY[@]}"; do
  i=$((i+1))
  for model in "${MODELS[@]}"; do

    if [ $SKIP_IF_FILE_FOUND -eq 1 ]; then
      if [ -f ${ROOT_FOLDER_PATH}/2_results/$eval/$model.jsonl ]; then
        echo "Skipping eval $eval for model $model because file already exists" >> $GENERAL_OUT
        continue
      fi
    fi
    echo "Running eval $eval for model $model" >> $GENERAL_OUT
    oaieval $model $eval \
      --max_samples $MAX_SAMPLES \
      --record_path=${ROOT_FOLDER_PATH}/2_results/$eval/$model.jsonl >> $IN_LIBRARY_EVALS_OUT 2>> $IN_LIBRARY_EVALS_ERR
  done
done
echo Run $i evals  >> $GENERAL_OUT
