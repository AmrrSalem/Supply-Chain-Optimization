# Supply Chain Optimization

An advanced **inventory optimization system** that applies multiple optimization algorithms to minimize costs while maintaining service levels — with an interactive **Streamlit** dashboard.

## Features

- **EOQ (Economic Order Quantity)** baseline model
- **Stochastic optimization** for demand uncertainty
- **Multi-product optimization** across SKUs
- **ML-based demand forecasting**
- Interactive **Streamlit + Plotly** dashboard
- Executive summary & PDF reporting

## Tech Stack

| Layer | Tools |
|-------|-------|
| Optimization | SciPy, NumPy, Pandas |
| Forecasting | scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Structure

```
├── sc_optimization.py    # Core optimization algorithms (EOQ, stochastic, multi-product)
├── streamlit_app.py      # Interactive Streamlit dashboard
├── requirements.txt
└── Supply Chain Optimization - Executive Summary & Implementation Guide.pdf
```

## Key Metrics Optimized

- Total inventory cost (holding + ordering + stockout)
- Reorder point & safety stock levels
- Service level targets
- Demand forecast accuracy
