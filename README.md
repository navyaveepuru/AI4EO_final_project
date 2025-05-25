# AI4EO_final_project
The final assignment (50%) for GEOL0069.

![Example of satellite, NDWI and water mask maps produced](https://github.com/user-attachments/assets/6c95d3fd-6b52-4242-ac18-8fbe62d4b67a)

## Description of the problem to be tackled

This project focuses on the application of machine learning techniques, a subset of artificial intelligence (AI), to the detection of inland water bodies in Bangladesh using satellite imagery. The motivation behind selecting this topic lies in the increasing global demand for near-real-time environmental monitoring, particularly in densely populated and climate-vulnerable regions (Islam et al, 2019). Inland water body detection has become a critical area of research due to the intensifying impacts of climate change, including rising sea levels, glacial melt, and more frequent extreme weather events, all of which contribute to abnormal and dangerous flooding patterns (IPCC, 2021; Setelle et al, 2014).

Bangladesh was chosen as the focal region due to its unique geographical and climatic conditions. Situated on the delta of the Ganges, Brahmaputra, and Meghna rivers, and being only a few meters above sea level, Bangladesh is home to one of the most extensive river networks in the world. This makes it especially prone to flooding, even under moderate rainfall conditions (Mirza, 2002). Furthermore, the low elevation and monsoonal climate amplify the risk of inland water expansion during both seasonal and anomalous rainfall events (Rahman and Salehin, 2013).

Adding to the complexity, Bangladesh is a developing country with limited economic resources and infrastructure for widespread, ground-based hydrological monitoring, which is traditionally expensive, labour-intensive, and requires specialist training (Rahman et al., 2019). With a population density exceeding 1,200 people per square kilometer (World Bank, 2022), the social and economic stakes of flood mismanagement are extremely high. Consequently, there is a pressing need for cost-effective, scalable, and automated solutions for water monitoring.

This is where machine learning and remote sensing offer transformative potential. By leveraging open-access satellite data—such as imagery from the Sentinel-2 constellation provided by the European Space Agency (ESA)—and combining it with supervised and unsupervised learning models, water bodies can be identified, segmented, and monitored in a timely and repeatable manner (Drusch et al., 2012). These models can rapidly process large volumes of image data, enabling near-real-time detection of water extent changes without the need for on-the-ground personnel. This not only reduces the operational cost of monitoring but also empowers early warning systems, urban planning, and climate adaptation efforts in resource-constrained regions like Bangladesh.

## Description of techniques used

### Sentinel-2 Imagery

<img width="532" alt="Image" src="https://github.com/user-attachments/assets/b4d3d0fc-f929-40f9-856f-7bc3e5ff763a" />

The Sentinel-2 mission, part of the European Union’s Copernicus Programme, consists of a pair of satellites—Sentinel-2A and Sentinel-2B—equipped with the MultiSpectral Instrument (MSI), which captures Earth’s surface in 13 spectral bands ranging from visible to short-wave infrared at varying spatial resolutions (10 m, 20 m, and 60 m) (Drusch et al., 2012). The high revisit frequency (every 5 days at the equator) and rich spectral data make Sentinel-2 especially suitable for environmental monitoring applications, including inland water detection. In this context, we use the bands Green (Band 3), Near-Infrared (Band 8), and Short-Wave Infrared (Band 11) to calculate the Normalized Difference Water Index (NDWI), which enhances the contrast between water bodies and other land features (McFeeters, 1996). The NDWI index can therefore be used to produce very accurate water masks (which detect which areas are water and which areas are land) for the region. This provides the basis to create both training data for the supervised models, and a 'correct' water mask for the test data that we can compare the model results to (ie. we are checking how close the model prediction is to the NDWI result; the closer the prediction, the more accurate the model is assessed to be).

### Supervised Learning: Logistic Regression and Random Forest

![Figure describing supervised learning techniques used](https://github.com/user-attachments/assets/c319162d-ddcb-4f10-b1c5-21153fcd8458)

Logistic regression is a supervised classification algorithm that models the probability of a binary outcome based on input features. In the context of inland water detection, the input features are Sentinel-2 RGB bands. This provides a satellite-like image to test the model. The logistic model uses the logistic (sigmoid) function to map the input feature space to a probability between 0 and 1, with a defined threshold (e.g., 0.5) used to classify pixels as water or non-water (Hosmer et al., 2013). This method is simple, interpretable, and computationally efficient. However, its performance is limited when the relationship between input features and the target class is nonlinear or more complex (Hosmer et al, 2013).

Random forest is an ensemble-based supervised learning algorithm that constructs a large number of decision trees during training and outputs the mode of their predictions for classification tasks (Breiman, 2001). Each decision tree is built from a random subset of training data and features, which helps reduce overfitting and improve generalization. In water body detection, random forest is well-suited to handle the complex, nonlinear relationships between spectral bands and water presence. For instance, subtle variations in vegetation or soil reflectance near riverbanks can be better captured through the ensemble voting mechanism, typically resulting in higher classification accuracy compared to simpler models like logistic regression (Belgiu and Drăguţ, 2016).

### Unsupervised Learning: K-Means Clustering

<img width="507" alt="Image" src="https://github.com/user-attachments/assets/c5f9e747-db90-4797-8fae-ab70bfe7accc" />

K-means is an unsupervised learning algorithm that partitions data into a predefined number of clusters by minimizing the variance within each cluster (MacQueen, 1967). When applied to satellite imagery, pixel values (e.g., RGB reflectance or NDWI values) are grouped into clusters based on spectral similarity.

For inland water body detection, a K-means model with 𝑘=2 can be used to segment pixels into water and non-water classes without requiring labeled training data. While less accurate than supervised methods due to lack of ground truth guidance, K-means can be useful in rapid assessments or regions lacking labeled datasets (Pekel et al, 2016).

## References

Belgiu, M. and Drăguţ, L., 2016. Random forest in remote sensing: A review of applications and future directions. ISPRS Journal of Photogrammetry and Remote Sensing, 114, pp.24–31. Available at: https://doi.org/10.1016/j.isprsjprs.2016.01.011

Breiman, L., 2001. Random forests. Machine Learning, 45(1), pp.5–32.

Canva, 2025. Canva [online graphic design tool]. Available at: https://www.canva.com [Accessed 18 May 2025].

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P., Martimort, P. and Meygret, A., 2012. Sentinel-2: ESA’s optical high-resolution mission for GMES operational services. Remote Sensing of Environment, 120, pp.25–36.

Hosmer, D.W., Lemeshow, S. and Sturdivant, R.X., 2013. Applied Logistic Regression. 3rd ed. Hoboken: Wiley.

IPCC, 2021. Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge: Cambridge University Press. Available at: https://www.ipcc.ch/report/ar6/wg1/ [Accessed 18 May 2025].

Islam, M.M., Barman, A., Kundu, G.K., Kabir, M.A. and Paul, B., 2019. Vulnerability of inland and coastal aquaculture to climate change: Evidence from a developing country. Aquaculture and Fisheries, 4(5), pp.183–189.

MacQueen, J., 1967. Some methods for classification and analysis of multivariate observations. In: Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability. Berkeley: University of California Press, pp.281–297.

McFeeters, S.K., 1996. The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing, 17(7), pp.1425–1432.

Mirza, M.M.Q., 2002. Global warming and changes in the probability of occurrence of floods in Bangladesh and implications. Global Environmental Change, 12(2), pp.127–138.

Pekel, J.F., Vancutsem, C., Bastin, L., Clerici, M., Vanbogaert, E., Bartholomé, E., Bonan, B., Cottam, A., Defourny, P. and Arino, O., 2016. High-resolution mapping of global surface water and its long-term changes. Nature, 540(7633), pp.418–422. Available at: https://doi.org/10.1038/nature20584

Rahman, M.M., Hassan, Q.K. and Rahman, M.S., 2019. Remote sensing-based seasonal variability of water bodies in Bangladesh. ISPRS Journal of Photogrammetry and Remote Sensing, 149, pp.53–66.

Rahman, R. and Salehin, M., 2013. Flood risks and reduction approaches in Bangladesh. In: Disaster Risk Reduction Approaches in Bangladesh. pp.65–90.

Settele, J., Scholes, R., Betts, R.A., Bunn, S., Leadley, P., Nepstad, D., Overpeck, J.T., Angel Taboada, M., Adrian, R., Allen, C. and Anderegg, W., 2014. Terrestrial and inland water systems. Cambridge University Press.

The World Bank, 2022. Population density (people per sq. km of land area) – Bangladesh. [online] Available at: https://data.worldbank.org/indicator/EN.POP.DNST [Accessed 18 May 2025].
