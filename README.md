# physical-ai-advanced1
Physical AI応用1講座 最終課題用のリポジトリ

各報酬等は公式サンプルに準拠
#実行用コード

#Google Colabで強化学習実行
%cd /content/Genesis-v1.4.5
!GENESIS_NO_RENDER=1 python go2_train_rough_rl.py --exp-name "go2_rough_rlfunction"

#Google Colabでlogのzipファイルをダウンロード

#ローカルPCで強化学習評価コード実行
python go2_eval_rough_rl.py -e go2_rough_rlfunction --ckpt 100
