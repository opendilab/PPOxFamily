import time
from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from threading import Thread
from sheep_env import SheepEnv

flask_app = Flask(__name__)
app = Api(
    app=flask_app,
    version="0.0.1",
    title="DI-sheep App",
    description="Play Sheep with Deep Reinforcement Learning, Powered by OpenDILab"
)

name_space = app.namespace('DI-sheep', description='DI-sheep APIs')
model = app.model(
    'DI-sheep params', {
        'command': fields.String(required=False, description="Command Field", help="reset, step"),
        'argument': fields.Integer(required=False, description="Argument Field", help="reset->level, step->action"),
    }
)
MAX_ENV_NUM = 50
ENV_TIMEOUT_SECOND = 60
envs = {}


def env_monitor():
    while True:
        cur_time = time.time()
        pop_keys = []
        for k, v in envs.items():
            if cur_time - v['update_time'] >= ENV_TIMEOUT_SECOND:
                pop_keys.append(k)
        for k in pop_keys:
            envs.pop(k)
        time.sleep(1)


app.env_thread = Thread(target=env_monitor, daemon=True)
app.env_thread.start()


@name_space.route("/")
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)
    def post(self):
        try:
            t_start = time.time()
            data = request.json
            cmd, arg, uid = data['command'], data['argument'], data['uid']
            ip = request.remote_addr
            ip = str(ip) + str(uid)

            if ip not in envs:
                if cmd == 'reset':
                    if len(envs) >= MAX_ENV_NUM:
                        response = jsonify(
                            {
                                "statusCode": 501,
                                "status": "No enough env resource, please wait a moment",
                            }
                        )
                        response.headers.add('Access-Control-Allow-Origin', '*')
                        return response
                    else:
                        env = SheepEnv(1, agent=False)
                        envs[ip] = {'env': env, 'update_time': time.time()}
                else:
                    response = jsonify(
                        {
                            "statusCode": 501,
                            "status": "No response for too long time, please reset the game",
                        }
                    )
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response
            else:
                env = envs[ip]['env']
                envs[ip]['update_time'] = time.time()
            if cmd == 'reset':
                env.reset(arg)
                scene = [item.to_json() for item in env.scene if item is not None]
                response = jsonify(
                    {
                        "statusCode": 200,
                        "status": "Execution action",
                        "result": {
                            "scene": scene,
                            "max_item_num": env.total_item_num,
                        }
                    }
                )
            elif cmd == 'step':
                _, _, done, _ = env.step(arg)
                scene = [item.to_json() for item in env.scene if item is not None]
                bucket = [item.to_json() for item in env.bucket]
                response = jsonify(
                    {
                        "statusCode": 200,
                        "status": "Execution action",
                        "result": {
                            "scene": scene,
                            "bucket": bucket,
                            "done": done,
                        }
                    }
                )
            else:
                response = jsonify({
                    "statusCode": 500,
                    "status": "Invalid command: {}".format(cmd),
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
            print('backend process time: {}'.format(time.time() - t_start))
            print('current env number: {}'.format(len(envs)))
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as e:
            import traceback
            print(repr(e))
            print(traceback.format_exc())
            response = jsonify({
                "statusCode": 500,
                "status": "Could not execute action",
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
