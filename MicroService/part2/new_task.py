import pika
import random

connection = pika.BlockingConnection(pika.ConnectionParameters('51.250.26.59',
                                                               5672,
                                                               '/',
                                                               pika.PlainCredentials('guest',
                                                                                     'guest123')))
channel = connection.channel()

# Declare the exchange
channel.exchange_declare(exchange='mesmes', exchange_type='fanout', durable=True)

# Publish a message to the exchange
message = "Check this %r" % (random.randint(0, 100))
channel.basic_publish(exchange='mesmes',
                      routing_key='',  # For fanout exchange, routing key is ignored
                      body=message,
                      properties=pika.BasicProperties(
                          delivery_mode=2,  # make message persistent
                      ))
print("Sent %r" % (message,))
connection.close()
