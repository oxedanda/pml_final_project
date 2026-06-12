# Project Proposal

## Project Title
Predicting Annual Wine Production by Viticultural Region in Portugal

## Project Category
Tabular data / regression

## Team Members
- Nº 27916 | Andrea Dombe
- Nº 27916 | Dandara França
- Nº 26298 | Fernanda Chácara

### Problem Statement
This project investigates how to predict annual wine production for Portuguese viticultural regions using historical official statistics from the Instituto da Vinha e do Vinho (IVV). This is an interesting problem because wine production is an important agricultural and economic indicator in Portugal, and it varies across regions and campaigns due to regional characteristics, changing production conditions, and year-to-year variability.

### Challenges
The main challenge of this project is the temporal structure of the data. Since wine production changes from one campaign to another, the evaluation strategy must avoid using information from future years when training the model. Another challenge is that the data are aggregated at the regional level, which limits the number of predictors directly available for modeling. This may reduce model complexity and make it harder to capture all the factors influencing production.

### Dataset
The main dataset will be the IVV table **“Evolução da Produção Total por Região Vitivinícola”**, available at [https://www.ivv.gov.pt/np4/163.html](https://www.ivv.gov.pt/np4/163.html). This dataset contains annual wine production values by viticultural region in hectoliters and will be collected directly from the IVV statistics portal, which is the official public source for these records in Portugal. If data integration is straightforward, the project may also use the IVV series **“Evolução da Área Total de Vinha - Portugal”**, available at [https://www.ivv.gov.pt/np4/10586.html](https://www.ivv.gov.pt/np4/10586.html), as an additional explanatory variable.

### Method or Algorithm
The proposed baseline method is **Linear Regression**, using year and viticultural region as the initial predictors. This model is suitable as a first approach because it is simple, transparent, and easy to interpret. If the dataset structure supports a slightly richer model without making the workflow too complex, vineyard area may be included as an additional feature, and a **Random Forest Regressor** may be tested as a comparison model.

### Evaluation
The results will be evaluated using standard regression metrics: **MAE**, **RMSE**, and **R²**. In addition to these global metrics, the analysis will compare model performance across regions and campaigns to better understand where the predictions perform well or poorly. The train, validation, and test organization will respect chronological order so that evaluation reflects a realistic forecasting setting and avoids information leakage.
