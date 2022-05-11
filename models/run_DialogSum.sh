export CUDA_VISIBLE_DEVICES=0

# ctrlsum
python run_gen.py --model_path ../pretrained_models/bart-base/ --dataset DialogSum --data_dir ../data/DialogSum/topic/ --output_dir DialogSum_bart_base_ctrlsum --batch_size 4 --max_source_length 1024
python run_predict.py --model_path output/DialogSum_bart_base_ctrlsum/checkpoint-xxx/ --dataset DialogSum --data_dir ../data/DialogSum/topic/ --output_dir DialogSum_bart_base_ctrlsum --batch_size 1 --max_source_length 1024

# aux1
python run_gen_aux1.py --model_path ../pretrained_models/bart-base/ --dataset DialogSum --data_dir ../data/DialogSum/topic/ --output_dir DialogSum_bart_base_ctrlsum_aux1 --max_source_length 1024
python run_predict_aux1.py --model_path output/DialogSum_bart_base_ctrlsum_aux1/checkpoint-xxx/ --dataset DialogSum --data_dir ../data/DialogSum/topic/ --output_dir DialogSum_bart_base_ctrlsum_aux1 --max_source_length 1024 --batch_size 1

# aux2
python process_data_DialogSum.py aux2
python run_gen_aux2.py --model_path ../pretrained_models/bart-base/ --dataset DialogSum --data_dir ../data/DialogSum/aux2/ --output_dir DialogSum_bart_base_ctrlsum_aux2 --max_source_length 1024  --epoch 10 --batch_size 2 --gradient_accumulation_steps 12
python run_predict_aux2.py --model_path output/DialogSum_bart_base_ctrlsum_aux2/checkpoint-xxx/ --dataset DialogSum --data_dir ../data/DialogSum/aux2/ --output_dir DialogSum_bart_base_ctrlsum_aux2 --max_source_length 1024 --batch_size 1

# aux3
python process_data_DialogSum.py aux3
python run_gen_aux3.py --model_path ../../pretrained_models/bart-base/ --dataset DialogSum --data_dir ../data/DialogSum/aux3/ --output_dir DialogSum_bart_base_ctrlsum_aux3--max_source_length 1024 --epoch 10
python run_predict_aux3.py --model_path output/DialogSum_bart_base_ctrlsum_aux3/checkpoint-xxx/ --dataset DialogSum --data_dir ../data/DialogSum/aux3/ --output_dir DialogSum_bart_base_ctrlsum_aux3 --max_source_length 1024 --batch_size 1


