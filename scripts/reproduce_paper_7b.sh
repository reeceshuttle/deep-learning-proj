export CUDA_VISIBLE_DEVICES=7;
python main.py --size=7b --step=50000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=100000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=150000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=200000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=250000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=300000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=350000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=400000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=450000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=500000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;
python main.py --size=7b --step=550000 --reproduce_paper=True --quantization_method=awq --dont_measure_losses;

python main.py --size=7b --step=50000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=100000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=150000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=200000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=250000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=300000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=350000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=400000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=450000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=500000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
python main.py --size=7b --step=550000 --reproduce_paper=True --quantization_method=naive --dont_measure_losses;
