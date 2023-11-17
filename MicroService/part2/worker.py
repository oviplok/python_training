import pika
import time

connection = pika.BlockingConnection(pika.ConnectionParameters('51.250.26.59',
                                                               5672,
                                                               '/',
                                                               pika.PlainCredentials('guest',
                                                                                     'guest123')))
channel = connection.channel()

channel.exchange_delete('mesmes')  # Delete the existing exchange
channel.exchange_declare(exchange='mesmes', exchange_type='fanout', durable=True)  # Re-declare the exchange with the desired durability

# Declare a non-exclusive, durable queue with a generated queue name
result = channel.queue_declare(queue='ikbo-12_egorov', exclusive=True)
queue_name = result.method.queue

# Bind the queue to the exchange
channel.queue_bind(exchange='mesmes', queue=queue_name)

print('Waiting for messages. To exit press CTRL+C')


def callback(ch, method, properties, body):
    print("Received %r" % (body,))
    time.sleep(body.count(b'.'))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=callback)

channel.start_consuming()
