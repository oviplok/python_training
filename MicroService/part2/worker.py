import pika
import time

connection = pika.BlockingConnection(pika.ConnectionParameters('51.250.26.59',
                                                               5672,
                                                               '/',
                                                               pika.PlainCredentials('guest',
                                                                                     'guest123')))
channel = connection.channel()
# Название эксклюзивной очереди
queue_name = 'ikbo-12_dmitrieva'

channel.queue_declare(queue=queue_name, durable=True)

print('Waiting for messages. To exit press CTRL+C')


def callback(ch, method, properties, body):
    print("Received %r" % (body,))
    time.sleep(body.count(10))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name,
                      on_message_callback=callback)

channel.start_consuming()
