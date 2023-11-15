import sys
import random

import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('51.250.26.59',
                                                               5672,
                                                               '/',
                                                               pika.PlainCredentials('guest',
                                                                                     'guest123')))
channel = connection.channel()
# Название эксклюзивной очереди
queue_name = 'ikbo-12_dmitrieva'

channel.queue_declare(queue=queue_name, durable=True)

message = "Check this%r" % (random.randint(0, 100))
channel.basic_publish(exchange='',
                      routing_key=queue_name,
                      body=message,
                      properties=pika.BasicProperties(
                          delivery_mode=2,  # make message persistent
                      ))
print("Sent %r" % (message,))
connection.close()
