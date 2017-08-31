from rq import Connection, Queue
from redis import Redis
from count_words import count_words_at_url # added import!
redis_conn = Redis()
q = Queue(connection=redis_conn)
job = q.enqueue(count_words_at_url, 'http://nvie.com')
print job
