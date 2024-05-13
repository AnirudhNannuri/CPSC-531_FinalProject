import csv
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from datetime import datetime

def parse_date(date_str):
    date_formats = ['%d-%m-%Y', '%Y-%m-%d']  # List of expected date formats
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    raise ValueError(f"Date format for {date_str} does not match expected formats.")

def import_data(session, file_path, insert_query, batch_size=100):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        batch = BatchStatement()
        record_count = 0

        try:
            for row in reader:
                row[1] = parse_date(row[1])  # Convert invoice_date
                # Specify the indices that need to be converted to int
                int_fields = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Indices for integer fields
                for field in int_fields:
                    row[field] = int(row[field])  # Convert fields to integers

                batch.add(insert_query, tuple(row))
                if len(batch) >= batch_size:
                    session.execute(batch)
                    batch.clear()
                    record_count += batch_size
                    print(f"{record_count} records inserted...")

            if batch:  # Execute any remaining records
                session.execute(batch)
                record_count += len(batch)
        except Exception as e:
            print(f"An error occurred while inserting data: {e}")
        finally:
            print(f"Total records inserted: {record_count}")

if __name__ == "__main__":
    cluster = Cluster(['localhost'])
    session = cluster.connect('frauddetection')
    insert_query = session.prepare("""INSERT INTO invoice_train (
            client_id, invoice_date, tarif_type, counter_number, counter_statue, counter_code,
            reading_remarque, counter_coefficient, consommation_level_1, consommation_level_2,
            consommation_level_3, consommation_level_4, old_index, new_index, months_number, counter_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    import_data(session, "/Users/csuftitan/Desktop/Coding/Big Data/FraudDetection/datasets/invoice_train.csv", insert_query, batch_size=100)
    session.shutdown()
    cluster.shutdown()
