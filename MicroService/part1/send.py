import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('51.250.26.59',
                                                               5672,
                                                               '/',
                                                               pika.PlainCredentials('guest',
                                                                                     'guest123')))
channel = connection.channel()
# Название эксклюзивной очереди
queue_name = 'ikbo-12_egorov'

#channel.queue_declare(queue=queue_name, exclusive=True,durable=True)

channel.basic_publish(exchange='',
                      routing_key=queue_name,
                      body=queue_name)
print("Sent %r" % queue_name)
connection.close()
