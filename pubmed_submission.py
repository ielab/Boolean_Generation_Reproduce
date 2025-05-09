from Bio import Entrez
import time
import pandas


min_date = "1975/01/01"
max_date = "2023/12/31"
email="xxx@gmail.com"
time_max= 10000
overall_max = 1000000
max_attempt_count = 20
def submission_one(query, current_min_date, current_max_date):
    attempt_count = 0
    while attempt_count < max_attempt_count:
        try:
            #print(query)
            handle = Entrez.esearch(db="pubmed", term=query, retmax=time_max, email=email, mindate=current_min_date,
                                    maxdate=current_max_date)
            #print(current_min_date)
            #print(current_max_date)
            record = Entrez.read(handle)
            #print(record)
            count = int(record["Count"])
            id_list = record["IdList"]
            return count, id_list
        except Exception as e:
            print(f"An error occurred: {e}")
            print(query)
            attempt_count += 1
            time.sleep(1)  # Wait for 5 seconds before retrying
    return 0, []

def pubmed_submission(query, dates, counter_too_many):
    original_chunks = [(dates_check(dates["mindate"]), dates_check(dates["maxdate"]))]
    id_lists = []
    #final_chunks = []
    print("Retrieving articles for the query: ", query)
    while len(original_chunks)>0:
        current_date_range_count, current_id_list = submission_one(query, original_chunks[0][0], original_chunks[0][1])
        print("Retrieved " + str(current_date_range_count) + " articles ", end=" ")
        if current_date_range_count > overall_max:
            counter_too_many += 1
            print("Retrieved " + str(current_date_range_count) + " articles ", end=" ")
            break
        if current_date_range_count > time_max:
            times = 3
            ts1 = pandas.Timestamp(original_chunks[0][0])
            ts2 = pandas.Timestamp(original_chunks[0][1])
            ading_date = (ts2-ts1)/times
            previous_mean = original_chunks[0][0]
            for index in range(1, times):
                chunk_mean = ts1 + (index*ading_date)
                chunk_mean = str(chunk_mean)[:10]
                chunk_now = [previous_mean, str(chunk_mean)]
                previous_mean = str(chunk_mean)
                original_chunks.append(chunk_now)
            chunk_last = [previous_mean, original_chunks[0][1]]
            original_chunks.append(chunk_last)
            original_chunks.pop(0)
        else:
            id_lists.extend(current_id_list)
            original_chunks.pop(0)
    print("Contain number of articles: ", len(set(id_lists)))
    return list(set(id_lists)), counter_too_many
def dates_check(date):
    if "/" not in date:
        new_date = date[:4] + "/" + date[4:6] + "/" + date[6:]
    else:
        # it need to also check if it's reversed, like 12/31/2023, then it should be 2023/12/31
        date_split = date.split("/")
        if len(date_split[0]) == 4:
            new_date = date
        else:
            new_date = date_split[2] + "/" + date_split[1] + "/" + date_split[0]
    return new_date