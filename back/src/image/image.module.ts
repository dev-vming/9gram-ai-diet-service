import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ImageService } from './image.service';
import { ImageController } from './image.controller';
import {Image} from "./entities/image.entity"
import { ImageRepository } from './repositories/image.repository';
import { User } from 'src/user/entities/user.entity';
import { UserModule } from 'src/user/user.module';
import { SplitImage } from './entities/splitImage.entity';
import { SplitImageRepository } from './repositories/splitImage.repository';
import { S3Client } from '@aws-sdk/client-s3';

@Module({
  imports: [TypeOrmModule.forFeature([Image, SplitImage])],
  providers: [ ImageService, ImageRepository, SplitImageRepository],
  controllers: [ ImageController ]
})
export class ImageModule {
}
