class Config(object):
    APPLICATION_ROOT = '/'


class DevelopmentConfig(Config):
    TESTING = True
    DEBUG = True
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SESSION_TIMEOUT_IN_MINUTES = 10


class ProductionConfig(Config):
    TESTING = False
    DEBUG = False
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_TIMEOUT_IN_MINUTES = 120


app_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}
