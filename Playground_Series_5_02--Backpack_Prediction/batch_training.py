for n_layers in 3 4; do # 2 5
	for layer_size in 32 64 128 256; do
		for drop_rate in 0.15; do # 0.1 0.35
			for learn_rate in 0.0001; do
				for epochs in 100; do
					nohup python3 tf_model.py ${n_layers} ${layer_size} ${drop_rate} ${learn_rate} ${epochs} &
				done
			done
		done
	done
done
