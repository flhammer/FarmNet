import pytest

from farmnet.data.wranglers import (
    to_farmnet,
    get_column_mapping,
    set_default_cfg,
)


class TestWrangler:
    set_default_cfg("tests/default.toml")

    def test_to_farmnet(self, raw_dataframe, farmnet_dataframe):
        column_mapping = get_column_mapping()
        with pytest.raises(TypeError):
            df = to_farmnet(raw_dataframe, column_mapping)

        df = to_farmnet(raw_dataframe, column_mapping=column_mapping)

        assert df.shape[1] == farmnet_dataframe.shape[1]
        assert set(df.columns).issubset(farmnet_dataframe.columns)
        assert df.index.name == farmnet_dataframe.index.name
