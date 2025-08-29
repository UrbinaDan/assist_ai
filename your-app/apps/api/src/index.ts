// Fastify bootstrapping placeholder
import Fastify from 'fastify';

const app = Fastify();

app.get('/health', async () => ({ status: 'ok' }));

app.listen({ port: 3001 }, err => {
  if (err) throw err;
});