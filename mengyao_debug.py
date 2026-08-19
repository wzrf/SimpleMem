import lancedb

db = lancedb.connect("./lancedb_data")

print(db.table_names())

table = db.open_table("memory_entries")

print("rows:", table.count_rows())
print(table.to_arrow().to_pylist()[:2])