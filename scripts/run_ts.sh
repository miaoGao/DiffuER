export PYTHONPATH=$PYTHONPATH:.

FILE_NAME=text_simple
DATA_NAME=text_simple

# model params
MODEL_NAME=/pretrained_models/bert-base-uncased
max_pos_len=128
tgt_len=128
TOKENIZER=$MODEL_NAME
DIFFU_SAMPLER=xy_uniform

# loss terms
USE_TEACHER=False
CONTEXT_AWARE=True         # tureusing bert last hidden as signal
LOGITS_AWARE=False
USE_AUX_AR_MODEL=False
RECONSTRUCT_TGT=x
USE_REG_L2=False
USE_REG_RANDOM=False

# train params
STEP=50000
warmup_steps=0
lr_step=$STEP   # The total number of training steps
batch_size=32
grad_accum=3
lr=8e-4
save_interval=10000
SEED=101
visible_device=0

# gen params
gen_timesteps=20
skip_timestep=100
load_step=$STEP
num_samples=20
EMA_RATE=0.9999

exp_root=my_output_ts/$DIFFU_SAMPLER\_$DATA_NAME\_aux_ar$USE_AUX_AR_MODEL\_bert_mse$USE_TEACHER\_seed$SEED\

# train
python ./train_utils/trainer_single_GPU.py \
model.name=$MODEL_NAME model.tokenizer=$TOKENIZER \
batch_size=$batch_size grad_accum=$grad_accum \
total_steps=$STEP warmup_steps=$warmup_steps exp.name=$FILE_NAME exp.root=$exp_root exp.seed=$SEED \
data.name=$DATA_NAME tgt_len=$tgt_len max_pos_len=$max_pos_len lr=$lr lr_step=$lr_step \
intermediate_size=512 num_attention_heads=8 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
eval_interval=3000 log_interval=1000 save_interval=$save_interval \
schedule_sampler=$DIFFU_SAMPLER time_att=True att_strategy='txl' use_AMP=True \
model.use_teacher=$USE_TEACHER model.teacher=$TOKENIZER model.context_aware=$CONTEXT_AWARE \
model.aux_ar_model=$USE_AUX_AR_MODEL model.reconstruct_tgt=$RECONSTRUCT_TGT \
visible_device=$visible_device

# gen
python ./gen_utils/m_generate.py \
model.name=$MODEL_NAME model.tokenizer=$TOKENIZER \
exp.name=$FILE_NAME exp.root=$exp_root load_step=$load_step \
data.name=$DATA_NAME max_pos_len=$max_pos_len num_samples=$num_samples \
intermediate_size=512 num_attention_heads=8 \
in_channels=64 out_channels=64 time_channels=64 \
skip_sample=True gen_timesteps=$gen_timesteps \
schedule_sampler=$DIFFU_SAMPLER time_att=True att_strategy='txl' \
tgt_len=$tgt_len prediction=True load_from_ema=True batch_size=64 \
model.use_teacher=$USE_TEACHER model.teacher=$TOKENIZER model.context_aware=$CONTEXT_AWARE \
model.aux_ar_model=$USE_AUX_AR_MODEL model.reconstruct_tgt=$RECONSTRUCT_TGT \
visible_device=$visible_device

# eval
if [ $DIFFU_SAMPLER == uniform ]; then
    SAMPLER_NAME=un
    folder=$exp_root/$DATA_NAME/$DATA_NAME/$STEP\_ema_0.9999_skip__$SAMPLER_NAME\_$skip_timestep
fi

if [ $DIFFU_SAMPLER == xy_uniform ]; then
    SAMPLER_NAME=xy
    folder=$exp_root/$DATA_NAME/$DATA_NAME/$STEP\_ema_0.9999_skip__$SAMPLER_NAME\_$gen_timesteps
fi

### select best from sampling results and prepare for files2rouge
python eval_utils/eval_utils.py --folder $folder

### Evaluating Rouge
ref_file=$folder/refs.txt
pred_file=$folder/preds.txt
files2rouge $pred_file $ref_file --ignore_empty_reference --ignore_empty_summary >> $exp_root/$DATA_NAME/$FILE_NAME/log.log

### Evaluating
python eval_utils/evaluation/eval.py --folder $folder --device_idx $visible_device >> $exp_root/$DATA_NAME/$FILE_NAME/log.log