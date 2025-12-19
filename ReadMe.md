# Bengaluru Food Clustering - K-Means

Clusters late-night food orders into 3 segments for delivery optimization.

## Dataset
Synthetic Bengaluru orders (1000 rows): `lat`, `lon`, `hour`(22-28), `value`(₹100-1500), `items`(1-10) [web:55]

## Pipeline
1. **Clean**: Remove outliers (z-score >3)
2. **Scale**: `StandardScaler()` on all features [web:38]
3. **Cluster**: `KMeans(n_clusters=3, init='k-means++', n_init=50)` [web:23]
4. **Validate**: Silhouette >0.55, Inertia <1300 [web:48]

## Key Plots
- `scatter(lon, lat, c=cluster)` - Geographic clusters
- Elbow curve (K=2-8)
- Silhouette analysis

## Results Target
| Metric | Good | Excellent |
|--------|------|-----------|
| Silhouette | 0.55 | 0.65+ |
| Inertia | 1350 | <1300 |

**Business**: 3 hubs (Koramangala snacks, Whitefield meals, Indiranagar premium) save ~25% routes [web:7]

## Concepts Demonstrated
- Scaling fixes distance bias
- K-Means++ beats random init
- Elbow + silhouette pick optimal K [web:21][web:56]
