export PYTHONPATH=$PYTHONPATH:.

# en -> de
visible_device=0
FILE_NAME=iwslt14_ende
DATA_NAME=iwslt14
STEP=200000
BSZ=16

MODEL_NAME=/pretrained_models/bert-base-uncased
USE_MBERT=True
diffusion_steps=2000
TGT_LEN=128
DIFFU_SAMPLER=uniform

# loss params
SEED=101
CONTEXT_AWARE=True
USE_AUX_AR_MODEL=False
AUX_AR_MODEL_LAYER=3
RECONSTRUCT_TGT=x
USE_TEACHER=False
TEACHER_MODEL=/pretrained_models/bert-base-multilingual-cased
TEACHER_WEIGHT=0.5
grad_accum=8
lr=5e-4

# gen params
gen_timesteps=20
load_step=$STEP
num_samples=10
EMA_RATE=0.9999
skip_timestep=100


exp_root=my_output_iwslt/$DIFFU_SAMPLER\_$DATA_NAME\_aux_ar$USE_AUX_AR_MODEL\_bert_mse$USE_TEACHER\_bsz$BSZ

python ./train_utils/trainer_single_GPU.py visible_device=$visible_device \
    model.name=$MODEL_NAME diffusion_steps=$diffusion_steps batch_size=$BSZ grad_accum=$grad_accum warmup_steps=0 \
    total_steps=$STEP exp.name=$FILE_NAME exp.root=$exp_root exp.seed=$SEED \
    data.name=$DATA_NAME tgt_len=$TGT_LEN max_pos_len=$TGT_LEN lr=$lr use_mbert=$USE_MBERT lr_step=$STEP \
    intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
    in_channels=64 out_channels=64 time_channels=64 \
    eval_interval=3000 log_interval=1000 \
    schedule_sampler=$DIFFU_SAMPLER time_att=True att_strategy='txl' use_AMP=True \
    src_lang='en' tgt_lang='de' \
    model.aux_ar_model=$USE_AUX_AR_MODEL model.reconstruct_tgt=$RECONSTRUCT_TGT model.aux_ar_model_layers=$AUX_AR_MODEL_LAYER \
    model.use_teacher=$USE_TEACHER model.teacher=$TEACHER_MODEL model.context_aware=$CONTEXT_AWARE model.teacher_weight=$TEACHER_WEIGHT

python ./gen_utils/m_generate.py visible_device=$visible_device \
    model.name=$MODEL_NAME diffusion_steps=$diffusion_steps batch_size=$BSZ \
    exp.name=$FILE_NAME load_step=$load_step exp.root=$exp_root \
    data.name=$DATA_NAME tgt_len=$TGT_LEN max_pos_len=$TGT_LEN use_mbert=$USE_MBERT num_samples=$num_samples \
    intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
    in_channels=64 out_channels=64 time_channels=64 \
    skip_sample=True gen_timesteps=$gen_timesteps skip_timestep=$skip_timestep \
    schedule_sampler=$DIFFU_SAMPLER time_att=True att_strategy='txl' \
    src_lang='en' tgt_lang='de' \
    model.aux_ar_model=$USE_AUX_AR_MODEL model.reconstruct_tgt=$RECONSTRUCT_TGT model.aux_ar_model_layers=$AUX_AR_MODEL_LAYER \
    model.use_teacher=$USE_TEACHER model.teacher=$TEACHER_MODEL model.context_aware=$CONTEXT_AWARE


if [ $DIFFU_SAMPLER == uniform ]; then
    SAMPLER_NAME=un
    folder=$exp_root/$DATA_NAME/$FILE_NAME/$STEP\_skip__$SAMPLER_NAME\_$skip_timestep
fi

if [ $DIFFU_SAMPLER == xy_uniform ]; then
    SAMPLER_NAME=xy
    folder=$exp_root/$DATA_NAME/$FILE_NAME/$STEP\_skip__$SAMPLER_NAME\_$gen_timesteps
fi

### select best from sampling results and prepare for files2rouge
python eval_utils/eval_utils.py --folder $folder

### Evaluating Rouge
ref_file=$folder/refs.txt
pred_file=$folder/preds.txt
files2rouge $pred_file $ref_file --ignore_empty_reference --ignore_empty_summary >> $exp_root/$DATA_NAME/$FILE_NAME/log.log

### Evaluating
python eval_utils/evaluation/eval.py --folder $folder --device_idx $visible_device >> $exp_root/$DATA_NAME/$FILE_NAME/log.log