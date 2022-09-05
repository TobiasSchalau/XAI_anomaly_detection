""" Tests for preprocessing class
"""

from numpy import sort
from xai_anomaly_detection.preprocessing import preprocessing

NSL_KDD_columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "outcome",
    "level",
]


def test_preprocessing():
    """Test function for testing preprocessing class"""
    target = preprocessing.PreprocessNSLKDD()

    assert all([a == b for a, b in zip(target.train_data.columns, NSL_KDD_columns)])
    assert all([a == b for a, b in zip(target.test_data.columns, NSL_KDD_columns)])

    categorial_columns = ["protocol_type", "service", "flag"]
    unique_values = {
        colname: target.train_data[colname].unique() for colname in categorial_columns
    }

    columns_preprocessed = NSL_KDD_columns
    for elem in categorial_columns:
        columns_preprocessed.remove(elem)

    for colname in unique_values:
        for new_col in [colname + "_" + val for val in unique_values[colname]]:
            columns_preprocessed.append(new_col)

    target.preprocessing()

    columns_preprocessed.sort()

    assert all(
        [a == b for a, b in zip(target.train_data.columns, columns_preprocessed)]
    )
    assert all([a == b for a, b in zip(target.test_data.columns, columns_preprocessed)])

    assert all([a == b for a, b in zip(sort(target.test_data['outcome'].unique()), ['attack', 'normal'])])
    assert all([a == b for a, b in zip(sort(target.train_data['outcome'].unique()), ['attack', 'normal'])])
