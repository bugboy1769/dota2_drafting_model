import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
from src.data import DataCollector
from src.utils import setup_logging

def main():
    #Load config
    with open('config.yaml', 'r') as f:
        config=yaml.safe_load(f)
    
    #Setup loading
    setup_logging()

    #Process data
    collector=DataCollector(config['path']['data_dir'])
    collector.process_matches(
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )

    print(f"\nData Processing Complete!")
    print(f"Processed data saved to: {collector.processed_dir}")

if __name__=='__main__':
    main()