# TAMU Datathon: DreamCity
A ML clustering algorithm to help users identify their dream city to live post COVID.

## Installation

Make sure you have python3 installed.

Download the following libraries:

```
pip install streamlit matplotlib seaborn plotly pandas sklearn
```

## Run

```
streamlit run DreamCity.py
```

## Features

From the sidebar, you can select either `By Preferences` or `Feeling Lucky` mode to conduct your search.

Both modes uses the K-means clustering algorithm to conduct the search.

### `By Preferences` Mode

In the sidebar, there are currently 9 available preferences that users can pick from.

Select at least one preference and adjust the slider that shows up on the right.

Press `Find` to begin your search.

A Plotly map will show up with cities of desired qualities, hover over the dots for detailed information.

You can change the preferences and press `Find` again to conduct a new search.


### `Feeling Lucky` Mode

You can input a city name in the given text field.

Press `Find` to begin your search.

A list of similar cities returned by the ML algorithm will be printed.

You can change the city and press `Find` again to conduct a new search.


