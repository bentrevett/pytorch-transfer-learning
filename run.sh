python predict_sentiment.py --data yelp --vocab yelp --name sentiment_yelp_no_pretrain_yelp_vocab
python predict_sentiment.py --data yelp --vocab amazon --name sentiment_yelp_no_pretrain_amazon_vocab
python predict_mask.py --data amazon --vocab amazon --name mask_amazon_amazon_vocab
python predict_mask.py --data amazon --vocab yelp --name mask_amazon_yelp_vocab
python predict_sentiment.py --data yelp --vocab amazon --name sentiment_mask_amazon_pretrain_amazon_vocab --load checkpoints/model-mask_amazon_amazon_vocab.pt
python predict_sentiment.py --data yelp --vocab yelp --name sentiment_mask_amazon_pretrain_yelp_vocab --load checkpoints/model-mask_amazon_yelp_vocab.pt