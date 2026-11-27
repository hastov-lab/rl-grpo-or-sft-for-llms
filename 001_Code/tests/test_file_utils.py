from file_utils import read_json, read_text


def test_file_readers(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"completed": true}')

    assert read_json(path) == {"completed": True}
    assert read_text(path) == '{"completed": true}'
