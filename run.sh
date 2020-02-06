python predict_sentiment.py --name predict_sentiment_no_pretrain
python predict_tags.py --name predict_tag_no_pretrain
python predict_mask.py --name predict_mask_no_pretrain
python predict_sentiment.py --name predict_sentiment_tag_pretrain --load checkpoints/model-predict_tag_no_pretrain
python predict_sentiment.py --name predict_sentiment_mask_pretrain --load checkpoints/model-predict_mask_no_pretrain