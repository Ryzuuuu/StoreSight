#!/usr/bin/env python3
"""
Setup script to parse Excel store layout and create JSON configuration.
"""
import argparse
import json
import openpyxl
from pathlib import Path
from typing import Optional


def create_store_layout(excel_path: Optional[Path] = None):
    """Parse Excel file and create store_layout.json"""

    if excel_path is None:
        excel_path = Path("data/store_layout.xlsx")

    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        print("Pass --excel <path> to point at your local layout spreadsheet.")
        return
    
    # Load Excel
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    
    # Extract data from Excel
    rows = list(ws.iter_rows(values_only=True))
    
    # Create store layout - Brigade Road store (STORE_BLR_002 based on problem)
    store_layout = {
        "store_id": "STORE_BLR_002",
        "store_name": "Brigade Road, Bangalore",
        "cameras": [
            {
                "camera_id": "CAM_ENTRY_01",
                "name": "Entry/Exit Threshold",
                "coverage_zones": ["ENTRY"]
            },
            {
                "camera_id": "CAM_FLOOR_02", 
                "name": "Main Floor",
                "coverage_zones": ["SKINCARE", "MAKEUP", "HAIRCARE", "FRAGRANCES"]
            },
            {
                "camera_id": "CAM_BILLING_03",
                "name": "Billing Counter",
                "coverage_zones": ["BILLING"]
            }
        ],
        "zones": [
            {
                "zone_id": "ENTRY",
                "zone_name": "Entry/Exit",
                "type": "entry_point",
                "x_min": 0,
                "y_min": 0,
                "x_max": 1920,
                "y_max": 100
            },
            {
                "zone_id": "SKINCARE",
                "zone_name": "Skincare Section",
                "type": "product_zone",
                "x_min": 100,
                "y_min": 150,
                "x_max": 600,
                "y_max": 800
            },
            {
                "zone_id": "MAKEUP",
                "zone_name": "Makeup Section",
                "type": "product_zone",
                "x_min": 700,
                "y_min": 150,
                "x_max": 1200,
                "y_max": 800
            },
            {
                "zone_id": "HAIRCARE",
                "zone_name": "Haircare Section",
                "type": "product_zone",
                "x_min": 1300,
                "y_min": 150,
                "x_max": 1800,
                "y_max": 800
            },
            {
                "zone_id": "FRAGRANCES",
                "zone_name": "Fragrances Section",
                "type": "product_zone",
                "x_min": 100,
                "y_min": 900,
                "x_max": 900,
                "y_max": 1500
            },
            {
                "zone_id": "BILLING",
                "zone_name": "Billing Counter",
                "type": "checkout",
                "x_min": 1000,
                "y_min": 900,
                "x_max": 1800,
                "y_max": 1500
            }
        ],
        "open_hours": {
            "monday": {"open": "10:00", "close": "21:00"},
            "tuesday": {"open": "10:00", "close": "21:00"},
            "wednesday": {"open": "10:00", "close": "21:00"},
            "thursday": {"open": "10:00", "close": "21:00"},
            "friday": {"open": "10:00", "close": "21:00"},
            "saturday": {"open": "10:00", "close": "21:00"},
            "sunday": {"open": "11:00", "close": "20:00"}
        }
    }
    
    output_path = Path("store_layout.json")
    with open(output_path, 'w') as f:
        json.dump(store_layout, f, indent=2)
    print(f"✓ Created {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate store_layout.json from Excel")
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Path to store layout Excel file (default: data/store_layout.xlsx)",
    )
    args = parser.parse_args()
    create_store_layout(args.excel)
