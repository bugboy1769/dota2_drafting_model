import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
from src.data import DataCollector
from src.utils import setup_logging

def main():
    parser=argparse.ArgumentParser(description='Collect Dota2 match data')
    parser.add_argument('--num-matches', type=int, default=10000, help='Number of matches to collect')
    parser.add_argument('--config', type=str, default='config.yaml')
    args=parser.parse_args()

    #Load config
    with open(args.config, 'r') as f:
        config=yaml.safe_load(f)
    
    #Setup logging
    log_file=Path(config['path']['log_dir'])/'collection.log'
    log_file.parent.mkdir(exist_ok=True)
    setup_logging(str(log_file))

    #Collect data
    collector=DataCollector(config['path']['data_dir'])
    collector.collect_matches(num_matches=args.num_matches)

    print(f"\n Data Collection Complete!")
    print(f"Raw data saved to: {collector.raw_dir}")

if __name__=='__main__':
    main()
